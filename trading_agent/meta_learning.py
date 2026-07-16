"""
trading_agent/meta_learning.py

Meta Learning module for the Trading Agent — offline historical
performance analysis only.

Responsibility (frozen):
    Read persisted trade history (written by trade_journal.py) and
    compute descriptive, per-symbol performance statistics from
    completed trades. Purely offline and read-only.

    Never influences the currently running trade. Never makes a
    trading decision. Never communicates with Kronos or the exchange.
    Its output is not wired into any live decision path in this
    codebase — it is a standalone analysis tool, consumed by a human
    or a future, explicitly-separate retraining/tuning process.

Explicitly NOT this module's responsibility:
    - Trade / hold / exit decision   -> trading_agent/decision_engine.py
    - Risk evaluation                  -> trading_agent/risk_manager.py
    - Capital allocation                -> trading_agent/capital_manager.py
    - Trade persistence                    -> trading_agent/trade_journal.py (this module only reads it)
    - Prediction Agent / Kronos work          -> never imported here
    - Exchange interaction                      -> never imported here
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from models.trading import TradeRecord

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = (
    "symbol", "entry_price", "exit_price", "amount", "pnl", "pnl_pct",
    "fee", "entry_time", "exit_time", "duration_min", "exit_reason", "new_balance",
)


class MetaLearningError(Exception):
    """Raised when trade history cannot be loaded or analyzed."""


@dataclass(frozen=True)
class SymbolPerformance:
    """
    Module-local — Meta Learning's output is never consumed by another
    Trading Agent module (it must never influence a live trade), so
    this stays local rather than being promoted to models/trading.py.

    performance_score direct-ports trading_bot.py's
    get_symbol_performance(): win_rate * (1 + avg_pnl / initial_balance),
    floored at 0.3.
    """
    symbol: str
    trade_count: int
    win_rate: float
    avg_pnl: float
    avg_pnl_pct: float
    avg_duration_min: float
    total_fee: float
    performance_score: float


def load_trade_history(path: Path) -> tuple[TradeRecord, ...]:
    """
    Reads trade history persisted by trade_journal.persist_trade_record()
    and reconstructs TradeRecord instances.

    Args:
        path: trade history CSV file.

    Returns:
        tuple[TradeRecord, ...], in file order.

    Raises:
        MetaLearningError: if the file is missing, unparseable, or
        missing required columns.
    """
    path = Path(path)
    if not path.exists():
        raise MetaLearningError(f"Trade history file not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise MetaLearningError(f"Failed to read trade history {path}: {exc}") from exc

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MetaLearningError(f"Trade history missing required columns: {missing}")

    if df.empty:
        logger.info("Meta learning: trade history at %s has no rows", path)
        return tuple()

    try:
        records = tuple(
            TradeRecord(
                symbol=row.symbol,
                entry_price=float(row.entry_price),
                exit_price=float(row.exit_price),
                amount=float(row.amount),
                pnl=float(row.pnl),
                pnl_pct=float(row.pnl_pct),
                fee=float(row.fee),
                entry_time=pd.Timestamp(row.entry_time).to_pydatetime(),
                exit_time=pd.Timestamp(row.exit_time).to_pydatetime(),
                duration_min=float(row.duration_min),
                exit_reason=str(row.exit_reason),
                new_balance=float(row.new_balance),
            )
            for row in df.itertuples(index=False)
        )
    except Exception as exc:
        raise MetaLearningError(f"Failed to reconstruct trade records from {path}: {exc}") from exc

    logger.info("Meta learning: loaded %d trade records from %s", len(records), path)
    return records


def analyze_symbol_performance(
    records: tuple[TradeRecord, ...],
    initial_balance: float,
) -> tuple[SymbolPerformance, ...]:
    """
    Computes per-symbol performance statistics from completed trades.

    Args:
        records: output of load_trade_history().
        initial_balance: Trading Agent's starting budget, used to
            normalize avg_pnl into performance_score — this module
            does not own capital, so it is supplied by the caller.

    Returns:
        tuple[SymbolPerformance, ...], one per distinct symbol present
        in records. Empty if records is empty.
    """
    if not records:
        logger.info("Meta learning: no trade records to analyze")
        return tuple()

    by_symbol: Dict[str, List[TradeRecord]] = {}
    for record in records:
        by_symbol.setdefault(record.symbol, []).append(record)

    results: List[SymbolPerformance] = []
    for symbol, trades in by_symbol.items():
        total = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        win_rate = wins / total
        avg_pnl = sum(t.pnl for t in trades) / total
        avg_pnl_pct = sum(t.pnl_pct for t in trades) / total
        avg_duration_min = sum(t.duration_min for t in trades) / total
        total_fee = sum(t.fee for t in trades)
        performance_score = max(0.3, win_rate * (1 + avg_pnl / initial_balance))

        results.append(SymbolPerformance(
            symbol=symbol,
            trade_count=total,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            avg_pnl_pct=avg_pnl_pct,
            avg_duration_min=avg_duration_min,
            total_fee=total_fee,
            performance_score=performance_score,
        ))

    logger.info("Meta learning: analyzed %d symbols from %d trades", len(results), len(records))
    return tuple(results)