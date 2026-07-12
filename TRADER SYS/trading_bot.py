"""
trading_bot.py

The Trading Bot — the project's runtime orchestrator ("operating
system"). Coordinates prediction_agent/runtime.py and
trading_agent/runtime.py. Owns runtime state and scheduling only.

Responsibility (frozen):
    - Start/stop the system.
    - Own runtime state: current artifact file locations and validity
      window, Forecast Cursor, open Position, available budget,
      Exchange instance, KronosWrapper instance, running flag.
    - Ensure a valid Prediction Artifact exists (request a new one only
      when none exists or the current one has expired — no additional
      Kronos inference while the current artifact remains valid).
    - Compute the Forecast Cursor from wall-clock time relative to the
      artifact's valid_from and timeframe.
    - Execute exactly one Trading Runtime cycle per loop iteration.
    - Schedule the next cycle at the next forecast-bar boundary rather
      than a fixed sleep interval, so the Forecast Cursor advances
      exactly one bar per cycle.

    It does NOT contain trading policy, risk logic, ranking, or any
    business logic — every decision still belongs to decision_engine.py,
    every risk computation to risk_manager.py, every prediction to the
    Prediction Agent, every exchange order to execution.py. This module
    only coordinates calls to prediction_agent.runtime.run_prediction_cycle()
    and trading_agent.runtime.run_trading_cycle().

Explicitly NOT this module's responsibility:
    - Any trading decision      -> trading_agent/decision_engine.py
    - Any risk computation        -> trading_agent/risk_manager.py
    - Any prediction algorithm      -> prediction_agent/
    - Any exchange order              -> trading_agent/execution.py
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from exchange import Exchange
from models.trading import Position

from prediction_agent.filters import FilterConfig
from prediction_agent.ranking import RankingConfig
from prediction_agent.downloader import DownloaderConfig
from prediction_agent.validator import ValidatorConfig
from prediction_agent.artifact_builder import ArtifactBuilderConfig
from prediction_agent.kronos_wrapper import KronosWrapper
from prediction_agent.runtime import run_prediction_cycle

from trading_agent.runtime import run_trading_cycle, TradingCycleResult

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class TradingBotConfig:
    """
    Bundles the static configuration TradingBot needs to construct
    Prediction Artifacts when the current one expires. Module-local —
    single consumer (TradingBot itself) — mirrors the same config-
    object pattern used throughout the Prediction Agent, not a new
    abstraction.
    """
    scanner_config: Dict[str, Any]
    filter_config: FilterConfig
    ranking_config: RankingConfig
    downloader_config: DownloaderConfig
    validator_config: ValidatorConfig
    artifact_builder_config: ArtifactBuilderConfig
    parquet_path: Path
    meta_path: Path
    trade_history_path: Path


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

        self._artifact_valid_from: Optional[datetime] = None
        self._artifact_valid_until: Optional[datetime] = None
        self._artifact_timeframe: Optional[str] = None

        self.running = False

    # ── Artifact lifecycle ──────────────────────────────────────────

    def _artifact_is_valid(self) -> bool:
        if self._artifact_valid_until is None:
            return False
        return datetime.utcnow() < self._artifact_valid_until

    def _ensure_valid_artifact(self) -> None:
        """
        Requests a new Prediction Artifact only if none exists yet or
        the current one has expired. Otherwise the existing artifact
        continues to be used — no additional Kronos inference while it
        remains valid, per the approved sequential-trading philosophy.
        """
        if self._artifact_is_valid():
            return

        logger.info("TradingBot: no valid artifact — running prediction cycle")
        artifact = run_prediction_cycle(
            self.exchange,
            self.kronos_wrapper,
            self.config.scanner_config,
            self.config.filter_config,
            self.config.ranking_config,
            self.config.downloader_config,
            self.config.validator_config,
            self.config.artifact_builder_config,
            self.config.parquet_path,
            self.config.meta_path,
        )
        self._artifact_valid_from = artifact.valid_from
        self._artifact_valid_until = artifact.valid_until
        self._artifact_timeframe = artifact.timeframe
        logger.info(
            "TradingBot: new artifact %s valid until %s",
            artifact.artifact_id, artifact.valid_until,
        )

    # ── Forecast Cursor ──────────────────────────────────────────────

    def _compute_cursor(self) -> int:
        """
        Forecast Cursor: current wall-clock time relative to
        artifact.valid_from, in units of the artifact's timeframe.
        Runtime state only — never stored in the artifact, Opportunity,
        or Position. Clamped to be non-negative.
        """
        assert self._artifact_valid_from is not None and self._artifact_timeframe is not None
        bar_seconds = _timeframe_to_seconds(self._artifact_timeframe)
        elapsed_seconds = (datetime.utcnow() - self._artifact_valid_from).total_seconds()
        cursor = int(elapsed_seconds // bar_seconds)
        return max(0, cursor)

    # ── Single cycle (independently testable) ────────────────────────

    def run_cycle(self) -> TradingCycleResult:
        """
        Executes exactly one complete runtime cycle: ensure a valid
        artifact, compute the cursor, run one Trading Runtime cycle,
        update runtime state. No loop, no sleep — safe to call directly
        in tests without invoking run()'s infinite loop.
        """
        self._ensure_valid_artifact()
        cursor = self._compute_cursor()

        result = run_trading_cycle(
            self.exchange,
            self.config.parquet_path,
            self.config.meta_path,
            cursor,
            self.open_position,
            self.available_budget,
            self.initial_balance,
            self.config.trade_history_path,
        )

        self.open_position = result.open_position
        self.available_budget = result.available_budget

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
        """
        self.running = True
        logger.info("TradingBot: starting")

        while self.running:
            try:
                self.run_cycle()
            except Exception:
                logger.exception("TradingBot: cycle failed, will retry next iteration")

            if self.running:
                self._sleep_until_next_bar()

        logger.info("TradingBot: stopped")

    def stop(self) -> None:
        """Signals run()'s loop to exit after the current wait/cycle."""
        self.running = False

    def _sleep_until_next_bar(self) -> None:
        bar_seconds = _timeframe_to_seconds(self._artifact_timeframe) if self._artifact_timeframe else 60
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