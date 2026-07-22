"""
trading_bot.py

The Trading Bot — the project's runtime orchestrator ("operating
system"). Coordinates prediction_agent/runtime.py and
trading_agent/runtime.py. Owns runtime state and scheduling only.

Responsibility (frozen):
    - Start/stop the system.
    - Own runtime state: tracked Prediction Artifacts (Entry Forecast +
      any Position Forecast still referenced by an open Position),
      Forecast Cursor, open Position, available budget, Exchange
      instance, KronosWrapper instance, running flag.
    - Ensure a valid Entry Forecast exists every cycle, unconditionally
      — prediction generation never pauses, regardless of whether a
      position is open (no additional Kronos inference while the
      current Entry Forecast remains valid, but refreshing it is never
      delayed by an open position either).
    - Distinguish Entry Forecast (newest artifact, used for evaluating
      new opportunities) from Position Forecast (the exact artifact
      that justified the currently open Position's entry, via
      Position.artifact_id — remains authoritative for that position's
      Forecast Cursor interpretation, windows, hold-time, and deviation
      checks until it closes). This prevents an open position from
      becoming orphaned when the Entry Forecast refreshes out from
      under it — see module-level comment near TradingBot for details.
    - Compute the Forecast Cursor from wall-clock time relative to
      whichever artifact (Entry or Position Forecast) governs the
      current cycle.
    - Execute exactly one Trading Runtime cycle per loop iteration.
    - Schedule the next cycle at the next forecast-bar boundary rather
      than a fixed sleep interval, so the Forecast Cursor advances
      exactly one bar per cycle.
    - Clean up artifact files no longer referenced by the Entry
      Forecast or an open Position, to bound disk usage under the new
      per-generation file scheme this requires.

    It does NOT contain trading policy, risk logic, ranking, or any
    business logic — every decision still belongs to decision_engine.py,
    every risk computation to risk_manager.py, every prediction to the
    Prediction Agent, every exchange order to execution.py. This module
    only coordinates calls to prediction_agent.runtime.run_prediction_cycle()
    and trading_agent.runtime.run_trading_cycle() — neither of which
    required any change to support Entry/Position Forecast separation,
    since both already operate on whichever artifact file paths they
    are given.

Explicitly NOT this module's responsibility:
    - Any trading decision      -> trading_agent/decision_engine.py
    - Any risk computation        -> trading_agent/risk_manager.py
    - Any prediction algorithm      -> prediction_agent/
    - Any exchange order              -> trading_agent/execution.py
"""

from __future__ import annotations

import json
import logging
import os
import signal
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional
from uuid import UUID, uuid4

from exchange import Exchange
from config.config import (
    ConfigError,
    configure_logging,
    ensure_runtime_directories,
    load_config,
    validate_exchange_connectivity,
)
from models.execution import OrderResult
from models.trading import Position, TradeRecord

from prediction_agent.filters import FilterConfig
from prediction_agent.ranking import RankingConfig
from prediction_agent.downloader import DownloaderConfig
from prediction_agent.validator import ValidatorConfig
from prediction_agent.artifact_builder import ArtifactBuilderConfig
from prediction_agent.kronos_wrapper import KronosWrapper
from prediction_agent.runtime import run_prediction_cycle

from trading_agent.artifact_loader import load_artifact, ArtifactLoaderError
from trading_agent.position_manager import close_position
from trading_agent.trade_journal import build_trade_record, persist_trade_record
from trading_agent.runtime import run_trading_cycle, TradingCycleResult

logger = logging.getLogger(__name__)


def _format_utc_and_local(dt: datetime) -> str:
    """Task 7 (logging UX) — see prediction_agent/artifact_builder.py's
    identical helper for the full rationale. UTC stays authoritative
    internally; this only affects human-readable log output."""
    utc_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    local_str = dt.replace(tzinfo=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    return f"{utc_str} UTC ({local_str} local)"

_TIMEFRAME_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400}


def _timeframe_to_seconds(timeframe: str) -> int:
    """
    Converts a ccxt-style timeframe string (e.g. "1m", "5m", "1h") into
    seconds. Same convention already used throughout the Prediction
    Agent (downloader.py, kronos_wrapper.py) — reimplemented locally
    (not imported from kronos_wrapper.py's private helper) to avoid
    depending on another module's private implementation detail.
    """
    unit_char = timeframe[-1]
    if unit_char not in _TIMEFRAME_UNIT_SECONDS:
        raise ValueError(f"Unsupported timeframe unit: {timeframe}")
    value = int(timeframe[:-1])
    return value * _TIMEFRAME_UNIT_SECONDS[unit_char]


class _ArtifactRef(NamedTuple):
    """
    Private, module-local bookkeeping record — file locations plus the
    validity metadata needed for Forecast Cursor computation, for one
    tracked Prediction Artifact generation. Deliberately a plain
    NamedTuple, not a new shared contract: it has no behavior of its
    own and nothing outside this module ever needs to reference it.
    """
    parquet_path: Path
    meta_path: Path
    valid_from: datetime
    valid_until: datetime
    timeframe: str


@dataclass(frozen=True)
class TradingBotConfig:
    """
    Bundles the static configuration TradingBot needs to construct
    Prediction Artifacts when the Entry Forecast expires. Module-local
    — single consumer (TradingBot itself) — mirrors the same config-
    object pattern used throughout the Prediction Agent, not a new
    abstraction.

    artifacts_dir replaces a single fixed parquet_path/meta_path pair:
    each prediction cycle now writes to a uniquely-named file pair
    within this directory, so an older artifact (still referenced by
    an open Position) is never overwritten by a newer one.

    position_state_path: single small JSON file recording the
    currently open Position, if any — the minimum persistence needed
    for restart recovery (see TradingBot._recover_state()). Written on
    ENTRY, deleted on EXIT.
    """
    scanner_config: Dict[str, Any]
    filter_config: FilterConfig
    ranking_config: RankingConfig
    downloader_config: DownloaderConfig
    validator_config: ValidatorConfig
    artifact_builder_config: ArtifactBuilderConfig
    artifacts_dir: Path
    trade_history_path: Path
    position_state_path: Path


class TradingBot:
    """
    The system orchestrator. Construct once; call run() for the
    continuous loop, or run_cycle() directly for a single, independently
    testable cycle.
    """

    def __init__(
        self,
        exchange: Exchange,
        kronos_wrapper: KronosWrapper,
        config: TradingBotConfig,
        initial_balance: float,
    ):
        self.exchange = exchange
        self.kronos_wrapper = kronos_wrapper
        self.config = config
        self.initial_balance = initial_balance

        self.available_budget = initial_balance
        self.open_position: Optional[Position] = None

        self._artifacts: Dict[UUID, _ArtifactRef] = {}
        self._current_artifact_id: Optional[UUID] = None
        self._recovered = False

        self.running = False

        # Session summary tracking (Task 3) — in-memory only, by design:
        # the summary is defined as "this runtime session only, not
        # historical statistics," so nothing here is persisted or reset
        # from a prior run. trade_history.csv/meta_learning.py remain
        # the durable, cross-session sources of truth.
        self._session_start = datetime.utcnow()
        self._prediction_cycles_run = 0
        self._artifacts_created_this_session = 0
        self._session_trades: List[TradeRecord] = []
        self._shutdown_reason = "graceful stop"

    # ── Artifact lifecycle (Entry Forecast) ──────────────────────────

    def _current_ref(self) -> Optional[_ArtifactRef]:
        if self._current_artifact_id is None:
            return None
        return self._artifacts.get(self._current_artifact_id)

    def _entry_forecast_is_valid(self) -> bool:
        ref = self._current_ref()
        if ref is None:
            return False
        return datetime.utcnow() < ref.valid_until

    def _ensure_valid_artifact(self) -> None:
        """
        Requests a new Entry Forecast only if none exists yet or the
        current one has expired — otherwise the existing Entry Forecast
        continues to be used, no additional Kronos inference while it
        remains valid. Runs unconditionally every cycle regardless of
        Position state: prediction generation never pauses to wait for
        a position to close.
        """
        if self._entry_forecast_is_valid():
            return

        logger.info("TradingBot: no valid Entry Forecast — running prediction cycle")
        generation_id = uuid4().hex
        parquet_path = self.config.artifacts_dir / f"artifact_{generation_id}.parquet"
        meta_path = self.config.artifacts_dir / f"artifact_{generation_id}.meta.json"

        artifact = run_prediction_cycle(
            self.exchange,
            self.kronos_wrapper,
            self.config.scanner_config,
            self.config.filter_config,
            self.config.ranking_config,
            self.config.downloader_config,
            self.config.validator_config,
            self.config.artifact_builder_config,
            parquet_path,
            meta_path,
        )

        self._artifacts[artifact.artifact_id] = _ArtifactRef(
            parquet_path=parquet_path,
            meta_path=meta_path,
            valid_from=artifact.valid_from,
            valid_until=artifact.valid_until,
            timeframe=artifact.timeframe,
        )
        self._current_artifact_id = artifact.artifact_id
        self._prediction_cycles_run += 1
        self._artifacts_created_this_session += 1
        logger.info(
            "TradingBot: new Entry Forecast %s valid until %s",
            artifact.artifact_id, _format_utc_and_local(artifact.valid_until),
        )

    def _cleanup_stale_artifacts(self) -> None:
        """
        Removes tracked artifacts that are neither the current Entry
        Forecast nor the open Position's Forecast — bounds disk usage
        under the per-generation file scheme. Safe to call every cycle;
        a no-op when nothing is stale.
        """
        referenced_ids = {self._current_artifact_id}
        if self.open_position is not None:
            referenced_ids.add(self.open_position.artifact_id)

        stale_ids = [aid for aid in self._artifacts if aid not in referenced_ids]
        for aid in stale_ids:
            ref = self._artifacts.pop(aid)
            for path in (ref.parquet_path, ref.meta_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("TradingBot: failed to remove stale artifact file %s: %s", path, exc)
            logger.info("TradingBot: cleaned up stale artifact %s", aid)

    # ── Forecast Cursor ──────────────────────────────────────────────

    def _compute_cursor(self, ref: _ArtifactRef) -> int:
        """
        Forecast Cursor: current wall-clock time relative to the
        governing artifact's valid_from, in units of that artifact's
        timeframe. Runtime state only — never stored in the artifact,
        Opportunity, or Position. Clamped to be non-negative.

        Which artifact "governs" depends on Position state — see
        run_cycle(): the Position Forecast while a position is open,
        the Entry Forecast otherwise. This is what keeps bar-index
        comparisons (profit_window, hold-time, forecast deviation)
        meaningful for an open position even after the Entry Forecast
        has since refreshed to a newer artifact. Needs no special
        recovery logic of its own after a restart: once the governing
        _ArtifactRef is recovered (see _recover_state()), correct
        cursor computation falls out of this same formula for free.
        """
        bar_seconds = _timeframe_to_seconds(ref.timeframe)
        elapsed_seconds = (datetime.utcnow() - ref.valid_from).total_seconds()
        cursor = int(elapsed_seconds // bar_seconds)
        return max(0, cursor)

    # ── Position persistence (minimum state needed for restart recovery) ──

    def _atomic_write_text(self, path: Path, text: str) -> None:
        """
        Writes `text` to `path` without ever leaving a partially-written
        file behind: stage to a temp file in the same directory (so the
        final os.replace() is a same-filesystem rename, not a copy),
        flush + fsync before the rename so the data is durable on disk
        before the old file is replaced, then atomically replace. Same
        commit pattern already used by prediction_agent/artifact_builder.py's
        _stage_temp_file()/save_artifact(), applied here to the single
        small JSON file that is the minimum state needed for restart
        recovery.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except Exception:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            raise

    def _persist_position(self, position: Position) -> None:
        data = asdict(position)
        data["entry_time"] = position.entry_time.isoformat()
        data["artifact_id"] = str(position.artifact_id)
        self._atomic_write_text(self.config.position_state_path, json.dumps(data))
        logger.info("TradingBot: persisted open position state for %s", position.symbol)

    def _load_persisted_position(self) -> Optional[Position]:
        path = self.config.position_state_path
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return Position(
                symbol=data["symbol"],
                entry_price=data["entry_price"],
                entry_time=datetime.fromisoformat(data["entry_time"]),
                amount=data["amount"],
                stop_loss=data["stop_loss"],
                take_profit=data["take_profit"],
                order_id=data["order_id"],
                fee=data["fee"],
                allocated_balance=data["allocated_balance"],
                remaining_balance=data["remaining_balance"],
                artifact_id=UUID(data["artifact_id"]),
                entry_bar=data["entry_bar"],
            )
        except Exception:
            logger.exception(
                "TradingBot: failed to parse persisted position state at %s — starting with no open position",
                path,
            )
            return None

    def _clear_persisted_position(self) -> None:
        self.config.position_state_path.unlink(missing_ok=True)

    def _reconcile_position_with_exchange(self, position: Position) -> Optional[Position]:
        """
        Startup reconciliation (Task 3): the persisted position-state
        file is a local cache, not the source of truth — the exchange
        is. This exists specifically to cover the window between "the
        exit order filled on the exchange" and "the local state file
        was successfully cleared": if clearing failed (disk error,
        crash, etc.) the stale file would otherwise resurrect an
        already-closed position on the next restart.

        Looks for a matching sell fill (same symbol, same amount within
        rounding tolerance, at or after entry_time) in exchange trade
        history. If found, the position is finalized exactly as
        run_trading_cycle()'s normal EXIT path does — journaled and
        budget reconciled — and None is returned so the caller treats
        this as no open position. If no such fill is found, the
        position is genuinely still open and is returned unchanged.

        This does not introduce a second persisted-state format —
        recovery always reads the same single Position JSON written by
        _persist_position(); this method only decides whether to trust
        it or override it with what the exchange reports.
        """
        since_ms = int(position.entry_time.timestamp() * 1000)
        try:
            fills = self.exchange.fetch_my_trades(position.symbol, since_ms)
        except Exception:
            logger.exception(
                "TradingBot: exchange reconciliation query failed for %s — "
                "trusting persisted state as still open",
                position.symbol,
            )
            return position

        tolerance = max(0.01 * position.amount, 1e-8)
        match = next(
            (
                fill for fill in fills
                if fill.side.lower() == "sell"
                and fill.timestamp_ms >= since_ms
                and abs(fill.amount - position.amount) <= tolerance
            ),
            None,
        )
        if match is None:
            return position

        logger.warning(
            "TradingBot: recovered position %s was already closed on the exchange "
            "(persisted state file was stale) — reconciling from trade history instead "
            "of resurrecting it",
            position.symbol,
        )

        # order_id is honestly unavailable here: TradeFill (from
        # fetch_my_trades) carries no order_id field (the same Stage 2
        # limitation execution.py's docstring already flags) — this
        # sentinel documents that fact rather than inventing one.
        synthetic_exit = OrderResult(
            order_id="reconciled-from-trade-history",
            symbol=match.symbol,
            side="sell",
            status="closed",
            filled_amount=match.amount,
            average_price=match.price,
            fee_cost=match.fee_cost,
        )

        try:
            close_position(position, synthetic_exit)
            record = build_trade_record(position, synthetic_exit, exit_reason="reconciled_on_recovery")
            persist_trade_record(record, self.config.trade_history_path)
            self.available_budget = record.new_balance
            self._session_trades.append(record)
        except Exception:
            # Same accepted residual risk already documented in
            # run_trading_cycle()'s own EXIT path: the exchange-side
            # fact (sold) is certain; only the bookkeeping is uncertain.
            pnl = (match.price - position.entry_price) * position.amount
            self.available_budget = position.remaining_balance + position.allocated_balance + pnl
            logger.exception(
                "TradingBot: reconciled sell for %s but journaling failed during recovery — "
                "position closed internally with budget reconciled to %.8f; this trade may be "
                "missing from trade_history and requires manual reconciliation",
                position.symbol, self.available_budget,
            )

        try:
            self._clear_persisted_position()
        except Exception:
            logger.exception(
                "TradingBot: failed to clear persisted position state for %s after recovery "
                "reconciliation — the stale file remains on disk. If this failure repeats across "
                "restarts, the same already-closed trade will be re-reconciled (and re-journaled) "
                "on each restart; a repeatedly failing clear here indicates a persistent disk/"
                "permissions problem requiring manual intervention",
                position.symbol,
            )

        return None

    # ── Restart recovery ─────────────────────────────────────────────

    def _scan_artifacts_dir(self):
        """
        Reads every meta/parquet pair in artifacts_dir in one pass.
        Returns (parsed, all_paths):
            parsed: Dict[UUID, _ArtifactRef] for files that parsed
                successfully — no separate persisted registry is kept;
                each meta.json already self-describes its own
                artifact_id via the existing, unmodified load_artifact().
            all_paths: every file discovered (parquet or meta.json,
                whether or not it parsed, whether or not it has a
                matching counterpart) — needed so the orphan sweep in
                _recover_state() can remove genuinely corrupt/unreadable
                files and mismatched single-file leftovers (e.g. a
                parquet committed but its meta.json crash-interrupted),
                not just successfully-parsed-but-unreferenced ones.
        """
        parsed: Dict[UUID, _ArtifactRef] = {}
        all_paths: set = set()
        if not self.config.artifacts_dir.exists():
            return parsed, all_paths

        meta_paths = {p.name: p for p in self.config.artifacts_dir.glob("*.meta.json")}
        parquet_paths = {p.name: p for p in self.config.artifacts_dir.glob("*.parquet")}
        all_paths.update(meta_paths.values())
        all_paths.update(parquet_paths.values())

        for meta_name, meta_path in sorted(meta_paths.items()):
            stem = meta_name[: -len(".meta.json")]
            parquet_path = parquet_paths.get(f"{stem}.parquet")
            if parquet_path is None:
                logger.warning("TradingBot: %s has no matching parquet file — will be treated as orphaned", meta_path)
                continue
            try:
                artifact = load_artifact(parquet_path, meta_path, enforce_expiry=False)
            except ArtifactLoaderError as exc:
                logger.warning("TradingBot: unreadable artifact file %s (will be treated as orphaned): %s", meta_path, exc)
                continue

            parsed[artifact.artifact_id] = _ArtifactRef(
                parquet_path=parquet_path,
                meta_path=meta_path,
                valid_from=artifact.valid_from,
                valid_until=artifact.valid_until,
                timeframe=artifact.timeframe,
            )

        return parsed, all_paths

    def _recover_state(self) -> None:
        """
        Runs once, lazily, on the first run_cycle() call — recovers
        runtime state from a previous, interrupted run. A no-op
        (beyond the orphan sweep) if no position was open at last
        shutdown (the common case).

        If a Position was open:
            - Restores it from the persisted JSON file.
            - Reconciles it against exchange trade history before
              trusting it (see _reconcile_position_with_exchange()) —
              the exchange, not the local file, is the source of truth
              for whether the position is genuinely still open.
            - Restores available_budget from Position.remaining_balance
              (the value already captured at entry — not re-derived) if
              reconciliation confirms the position is still open, or
              from the reconciled trade's outcome if it was not.
            - Locates and recovers the Position Forecast's _ArtifactRef
              from the directory scan.
            - Restores no Forecast Cursor state directly — it is always
              recomputed fresh from wall-clock time, so recovering the
              _ArtifactRef alone is sufficient (see _compute_cursor()).

        Also recovers the most recently generated, still-valid artifact
        (other than the Position Forecast) as the Entry Forecast, if
        one exists — avoids an unnecessary Kronos call immediately
        after restart when a perfectly good, not-yet-expired Entry
        Forecast already sits on disk from before the crash. There is
        no signal in an artifact file itself saying "I was the Entry
        Forecast" (vs. some other still-valid artifact), so the most
        recently generated non-position candidate is used as a
        reasonable, explicit heuristic — not a hidden assumption.

        Any remaining files on disk — including ones that failed to
        parse at all (corrupt/unreadable) or have no matching
        counterpart (a lone parquet or meta.json from a crash mid-
        commit) — are orphans from an interrupted prior run and are
        removed here, so they don't accumulate indefinitely across
        repeated restarts over a long-running deployment.

        If the Position Forecast specifically cannot be located on
        disk (files deleted/corrupted — a genuine failure, not the
        normal path), no new code is needed to handle it: run_cycle()'s
        existing defensive fallback and decision_engine.py's own
        artifact_id-mismatch safety net (both already in place and
        tested) apply exactly as they would for any other stale-
        artifact case.
        """
        loaded_position = self._load_persisted_position()
        if loaded_position is not None:
            self.open_position = self._reconcile_position_with_exchange(loaded_position)
        else:
            self.open_position = None
        disk_artifacts, all_paths = self._scan_artifacts_dir()
        keep_ids: set = set()

        if self.open_position is not None:
            logger.info(
                "TradingBot: recovered open position %s (artifact_id=%s) from a previous run",
                self.open_position.symbol, self.open_position.artifact_id,
            )
            self.available_budget = self.open_position.remaining_balance

            ref = disk_artifacts.get(self.open_position.artifact_id)
            if ref is not None:
                self._artifacts[self.open_position.artifact_id] = ref
                keep_ids.add(self.open_position.artifact_id)
                logger.info("TradingBot: Position Forecast recovered from %s", ref.parquet_path)
            else:
                logger.error(
                    "TradingBot: could not locate Position Forecast %s on disk for recovered position %s — "
                    "will fall back to safety-only evaluation until horizon exhaustion",
                    self.open_position.artifact_id, self.open_position.symbol,
                )
        else:
            logger.info("TradingBot: no persisted position found — starting fresh")

        now = datetime.utcnow()
        entry_candidates = [
            (aid, ref) for aid, ref in disk_artifacts.items()
            if aid not in keep_ids and ref.valid_until > now
        ]
        if entry_candidates:
            entry_id, entry_ref = max(entry_candidates, key=lambda pair: pair[1].valid_from)
            self._artifacts[entry_id] = entry_ref
            self._current_artifact_id = entry_id
            keep_ids.add(entry_id)
            logger.info("TradingBot: recovered still-valid Entry Forecast %s from a previous run", entry_id)

        kept_paths: set = set()
        for aid in keep_ids:
            ref = self._artifacts[aid]
            kept_paths.add(ref.parquet_path)
            kept_paths.add(ref.meta_path)

        for path in all_paths:
            if path in kept_paths:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("TradingBot: failed to remove orphaned artifact file %s: %s", path, exc)
            else:
                logger.info("TradingBot: removed orphaned artifact file %s found during recovery", path)

    # ── Single cycle (independently testable) ────────────────────────

    def run_cycle(self) -> TradingCycleResult:
        """
        Executes exactly one complete runtime cycle: recover prior
        runtime state on the first call only, ensure a valid Entry
        Forecast (always), select the governing artifact for this
        cycle (Position Forecast if a position is open, Entry Forecast
        otherwise), compute the cursor relative to it, run one Trading
        Runtime cycle, update runtime state (including persisting or
        clearing the open-position state file), clean up stale
        artifacts. No loop, no sleep — safe to call directly in tests
        without invoking run()'s infinite loop.
        """
        if not self._recovered:
            self._recover_state()
            self._recovered = True

        self._ensure_valid_artifact()

        if self.open_position is not None:
            ref = self._artifacts.get(self.open_position.artifact_id)
            if ref is None:
                # Defensive only — should not happen, since a
                # referenced artifact is never pruned by
                # _cleanup_stale_artifacts(). Fall back to the Entry
                # Forecast; decision_engine.py's own artifact_id check
                # will correctly treat the position as having no
                # timeline reference and fall back to safety-only
                # evaluation, same as any other stale-artifact case.
                logger.error(
                    "TradingBot: open position references untracked artifact %s — falling back to Entry Forecast",
                    self.open_position.artifact_id,
                )
                ref = self._current_ref()
        else:
            ref = self._current_ref()

        assert ref is not None  # guaranteed by _ensure_valid_artifact() having just run
        cursor = self._compute_cursor(ref)

        previous_position = self.open_position

        result = run_trading_cycle(
            self.exchange,
            ref.parquet_path,
            ref.meta_path,
            cursor,
            self.open_position,
            self.available_budget,
            self.initial_balance,
            self.config.trade_history_path,
        )

        # Persist BEFORE updating in-memory state: closes the crash
        # window between "a fill is confirmed" and "that fact survives
        # a restart." Wrapped so a persistence I/O failure degrades
        # gracefully (this run keeps managing the position correctly in
        # memory; it just won't survive a restart until the next
        # successful persist) rather than aborting the rest of this
        # cycle's already-completed trade.
        if previous_position is None and result.open_position is not None:
            try:
                self._persist_position(result.open_position)
            except Exception:
                logger.exception(
                    "TradingBot: failed to persist position state for %s — in-memory state remains "
                    "correct for this run, but this position will not survive a restart until the "
                    "next successful persist",
                    result.open_position.symbol,
                )
        elif previous_position is not None and result.open_position is None:
            try:
                self._clear_persisted_position()
            except Exception:
                logger.exception("TradingBot: failed to clear persisted position state")

        self.open_position = result.open_position
        self.available_budget = result.available_budget

        if result.trade_record is not None:
            self._session_trades.append(result.trade_record)

        self._cleanup_stale_artifacts()

        logger.info(
            "TradingBot: cycle complete — cursor=%d action=%s budget=%.4f position=%s",
            cursor, result.decision.action,
            self.available_budget, self.open_position.symbol if self.open_position else None,
        )
        return result

    # ── Continuous loop ───────────────────────────────────────────────

    def run(self) -> None:
        """
        Owns the infinite loop: repeatedly calls run_cycle(), then
        sleeps until the next forecast-bar boundary (not a fixed
        interval) so the Forecast Cursor advances exactly one bar per
        cycle. Interruptible by self.running at short intervals, so a
        shutdown request is honored promptly rather than waiting out a
        full bar.

        A session summary (Task 3) is always logged when this method
        returns or raises — normal exit, a signal-triggered stop(), or
        a genuinely unhandled exception escaping the loop all produce
        one, via the try/finally below. Per-cycle exceptions do NOT
        reach here at all (see the inner try/except) — the loop
        already retries those on the next iteration, so only something
        that escapes run_cycle()'s own handling counts as "unhandled"
        for the summary's shutdown reason.
        """
        self.running = True
        logger.info("TradingBot: starting")

        try:
            while self.running:
                try:
                    self.run_cycle()
                except Exception:
                    logger.exception("TradingBot: cycle failed, will retry next iteration")

                if self.running:
                    self._sleep_until_next_bar()
        except BaseException as exc:
            self._shutdown_reason = f"unhandled exception: {type(exc).__name__}: {exc}"
            raise
        finally:
            logger.info("TradingBot: stopped")
            self._log_session_summary()

    def stop(self, reason: str = "stop() called") -> None:
        """Signals run()'s loop to exit after the current wait/cycle."""
        self._shutdown_reason = reason
        self.running = False

    def install_signal_handlers(self) -> None:
        """
        Registers SIGINT/SIGTERM to call stop(), so an external
        shutdown request (Ctrl+C, `docker stop`, systemd stop, etc.)
        triggers the existing graceful-shutdown path — finish the
        current cycle, then exit the loop — rather than an abrupt
        KeyboardInterrupt/SIGTERM traceback mid-cycle.

        A signal arriving while run_cycle() is already in progress does
        NOT abort it: self.running is only checked between cycles and
        during the interruptible bar-boundary sleep (already polls
        every ~1s). This is intentional, not a limitation — a cycle
        already executing an ENTRY or EXIT should always be allowed to
        finish (including persistence) rather than being cut off
        mid-transaction.

        Opt-in, not called automatically by __init__ or run() —
        installing process-wide signal handlers as a side effect of
        construction would be a surprising hidden effect. The caller
        (e.g. a __main__ entry point) explicitly opts in.
        """
        def _handle_signal(signum, frame):
            logger.info("TradingBot: received signal %s — stopping after current cycle", signum)
            self.stop(reason=f"signal {signum} received (Ctrl+C or termination request)")

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    def _log_session_summary(self) -> None:
        """
        Task 3: a concise report of THIS runtime session only — not
        historical/cross-session statistics (those already live in
        trade_history.csv / meta_learning.py). Single log call, so it
        lands wherever configure_logging() already sends output (file
        + stdout) with no new persisted format to maintain.
        """
        end_time = datetime.utcnow()
        duration = end_time - self._session_start

        ending_balance = self.available_budget
        net_pnl = ending_balance - self.initial_balance
        net_return_pct = (net_pnl / self.initial_balance * 100) if self.initial_balance else 0.0

        total_trades = len(self._session_trades)
        wins = sum(1 for t in self._session_trades if t.pnl > 0)
        losses = sum(1 for t in self._session_trades if t.pnl < 0)
        win_rate = (wins / total_trades * 100) if total_trades else 0.0
        total_fees = sum(t.fee for t in self._session_trades)

        if self.open_position is not None:
            open_position_note = (
                f"{self.open_position.symbol} still open (entry={self.open_position.entry_price:.8f}, "
                f"amount={self.open_position.amount:.8f}) — NOT included in ending balance above, "
                f"since it is unrealized"
            )
        else:
            open_position_note = "none"

        logger.info(
            "\n" + "=" * 60 +
            "\nTRADING SESSION SUMMARY\n" + "=" * 60 +
            f"\nSession"
            f"\n  Start:              {_format_utc_and_local(self._session_start)}"
            f"\n  End:                {_format_utc_and_local(end_time)}"
            f"\n  Duration:           {duration}"
            f"\nCapital"
            f"\n  Starting balance:   {self.initial_balance:.4f}"
            f"\n  Ending balance:     {ending_balance:.4f}"
            f"\n  Net PnL:            {net_pnl:+.4f}"
            f"\n  Net Return:         {net_return_pct:+.2f}%"
            f"\n  Open position:      {open_position_note}"
            f"\nTrading"
            f"\n  Trades executed:    {total_trades}"
            f"\n  Wins:               {wins}"
            f"\n  Losses:             {losses}"
            f"\n  Win rate:           {win_rate:.1f}%"
            f"\nFees"
            f"\n  Total fees paid:    {total_fees:.8f}"
            f"\nPrediction"
            f"\n  Prediction cycles:  {self._prediction_cycles_run}"
            f"\n  Artifacts created:  {self._artifacts_created_this_session}"
            f"\nRuntime"
            f"\n  Shutdown reason:    {self._shutdown_reason}"
            + "\n" + "=" * 60
        )

    def _sleep_until_next_bar(self) -> None:
        ref = self._artifacts.get(self.open_position.artifact_id) if self.open_position else self._current_ref()
        bar_seconds = _timeframe_to_seconds(ref.timeframe) if ref else 60
        now = datetime.utcnow()
        seconds_into_bar = now.timestamp() % bar_seconds
        remaining = bar_seconds - seconds_into_bar

        # Sleep in short increments so `running` can interrupt promptly
        # rather than blocking for up to a full bar interval.
        poll_interval = 1.0
        slept = 0.0
        while self.running and slept < remaining:
            time.sleep(min(poll_interval, remaining - slept))
            slept += poll_interval


# =============================================================================
#  Deployment startup sequence
#
#  This is the project's sole entry point (`python trading_bot.py`) — no
#  separate bootstrap/launcher script exists. Everything below is pure
#  orchestration/wiring: load configuration, validate it (including
#  exchange connectivity), construct Exchange/KronosWrapper/TradingBot,
#  and start the run loop. No trading, prediction, or business logic
#  lives here — every decision still belongs to decision_engine.py,
#  every risk computation to risk_manager.py, every prediction to the
#  Prediction Agent — this only constructs and wires them together,
#  which is exactly TradingBot's own documented "Start/stop the
#  system" responsibility (see module docstring above).
# =============================================================================

def main() -> None:
    # Minimal, unconfigured logging so a configuration error itself is
    # visible before configure_logging() (which needs a valid config)
    # can run.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        config = load_config()
    except ConfigError as exc:
        logger.critical("Startup aborted — invalid configuration:\n%s", exc)
        raise SystemExit(1) from exc

    try:
        ensure_runtime_directories(config)
    except ConfigError as exc:
        logger.critical("Startup aborted — could not prepare runtime directories: %s", exc)
        raise SystemExit(1) from exc

    # Reconfigure logging now that a valid, directory-backed log_file
    # is available — replaces the minimal handler set above.
    configure_logging(config)

    exchange = Exchange(config.exchange_id, config.binance_api_key, config.binance_api_secret)

    try:
        validate_exchange_connectivity(exchange)
    except ConfigError as exc:
        logger.critical("Startup aborted — exchange connectivity check failed: %s", exc)
        raise SystemExit(1) from exc

    logger.info("TradingBot: exchange credentials verified (exchange_id=%s)", config.exchange_id)

    kronos_wrapper = KronosWrapper(config.kronos_wrapper_config)

    bot_config = TradingBotConfig(
        scanner_config=config.scanner_config,
        filter_config=config.filter_config,
        ranking_config=config.ranking_config,
        downloader_config=config.downloader_config,
        validator_config=config.validator_config,
        artifact_builder_config=config.artifact_builder_config,
        artifacts_dir=config.artifacts_dir,
        trade_history_path=config.trade_history_path,
        position_state_path=config.position_state_path,
    )

    bot = TradingBot(
        exchange=exchange,
        kronos_wrapper=kronos_wrapper,
        config=bot_config,
        initial_balance=config.initial_balance,
    )
    bot.install_signal_handlers()
    bot.run()


if __name__ == "__main__":
    main()