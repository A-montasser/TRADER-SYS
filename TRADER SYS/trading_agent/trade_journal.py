"""
trading_agent/trade_journal.py

Trade Journal module for the Trading Agent.

Responsibility (frozen):
    Construct a TradeRecord (models/trading.py) from a closed Position
    and its exit OrderResult, and persist it as the trade history
    source of truth. Persistence only — no trading decisions, no risk
    computation, no capital allocation, no Kronos, no exchange
    interaction beyond writing the record to disk.

Explicitly NOT this module's responsibility:
    - Trade / hold / exit decision   -> trading_agent/decision_engine.py
    - Risk evaluation                  -> trading_agent/risk_manager.py
    - Capital allocation                -> trading_agent/capital_manager.py
    - Position lifecycle                  -> trading_agent/position_manager.py
    - Order execution                       -> trading_agent/execution.py
    - Prediction Agent work                    -> never imported here
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from models.execution import OrderResult
from models.trading import Position, TradeRecord

logger = logging.getLogger(__name__)


class TradeJournalError(Exception):
    """Raised when a trade record cannot be built or persisted."""


def build_trade_record(
    position: Position,
    exit_order_result: OrderResult,
    exit_reason: str,
) -> TradeRecord:
    """
    Constructs a TradeRecord from a closed Position and its exit order.

    new_balance is computed internally from position.remaining_balance +
    position.allocated_balance + pnl — this module already has every
    input needed (position, exit_order_result) and already computes pnl
    for the record itself, so deriving new_balance here too avoids the
    same formula being duplicated by the caller (previously
    trading_agent/runtime.py recomputed pnl independently just to pass
    new_balance in — a genuine responsibility-leakage bug, not a
    deliberate design; fixed here).

    Total fee (position.fee + exit_order_result.fee_cost) is recorded
    as the single round-trip cost — a deliberate choice: legacy
    trading_bot.py's exit_trade() trade_result recorded exit fee only,
    but TradeRecord.fee has no fixed formula attached to it, and the
    combined figure is a more complete accounting within the same
    single field.

    exit_reason is caller-supplied rather than computed: this module
    does not make decisions, so it cannot derive it itself.

    Args:
        position: the closed Position.
        exit_order_result: the filled sell order that closed it.
        exit_reason: human-readable reason the position was closed
            (e.g. a TradeDecision.reason string from decision_engine.py).

    Returns:
        TradeRecord

    Raises:
        TradeJournalError: if exit_order_result does not correspond to
        this position (symbol mismatch or not a sell).
    """
    if exit_order_result.symbol != position.symbol:
        raise TradeJournalError(
            f"Exit order symbol {exit_order_result.symbol!r} does not match "
            f"position symbol {position.symbol!r}"
        )
    if exit_order_result.side != "sell":
        raise TradeJournalError(
            f"Exit order for {position.symbol} has side={exit_order_result.side!r}, expected 'sell'"
        )

    exit_price = exit_order_result.average_price
    exit_time = datetime.utcnow()
    pnl = (exit_price - position.entry_price) * position.amount
    pnl_pct = (exit_price / position.entry_price - 1) * 100
    duration_min = (exit_time - position.entry_time).total_seconds() / 60
    total_fee = position.fee + exit_order_result.fee_cost
    new_balance = position.remaining_balance + position.allocated_balance + pnl

    record = TradeRecord(
        symbol=position.symbol,
        entry_price=position.entry_price,
        exit_price=exit_price,
        amount=position.amount,
        pnl=pnl,
        pnl_pct=pnl_pct,
        fee=total_fee,
        entry_time=position.entry_time,
        exit_time=exit_time,
        duration_min=duration_min,
        exit_reason=exit_reason,
        new_balance=new_balance,
    )

    logger.info(
        "Trade record built: %s pnl=%.8f (%.4f%%) duration=%.2fmin reason=%s new_balance=%.8f",
        record.symbol, record.pnl, record.pnl_pct, record.duration_min, record.exit_reason, record.new_balance,
    )
    return record


def persist_trade_record(record: TradeRecord, path: Path) -> None:
    """
    Appends one TradeRecord as a row to the trade history file at
    `path`, creating it with headers if it does not yet exist.

    Args:
        record: the TradeRecord to persist.
        path: destination file (caller-supplied — this module has no
            hardcoded default path).

    Raises:
        TradeJournalError: if the record cannot be written.
    """
    path = Path(path)
    try:
        file_exists = path.exists()
        row_df = pd.DataFrame([asdict(record)])
        path.parent.mkdir(parents=True, exist_ok=True)
        row_df.to_csv(path, mode="a", header=not file_exists, index=False)
    except Exception as exc:
        raise TradeJournalError(f"Failed to persist trade record to {path}: {exc}") from exc

    logger.info("Trade record persisted: %s -> %s", record.symbol, path)