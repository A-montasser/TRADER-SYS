"""
trading_agent/position_manager.py

Position Lifecycle module for the Trading Agent.

Responsibility (frozen):
    Own the lifecycle of the Trading Agent's single open Position:
    create it from a successful order execution, and close it when
    decision_engine.py has already decided EXIT and execution has
    already filled the exit order. Never decides when to enter or
    exit — it only reflects decisions already made by decision_engine.py
    and execution results.

Explicitly NOT this module's responsibility:
    - Trade / no-trade / hold / exit decision -> trading_agent/decision_engine.py
    - Prediction quality evaluation              -> trading_agent/scenario_analyzer.py
    - Risk evaluation                             -> trading_agent/risk_manager.py
    - Capital allocation                           -> trading_agent/capital_manager.py
    - Order execution                               -> trading_agent/execution.py
    - Trade history / journaling                     -> trading_agent/trade_journal.py

Implementation constraints (flagged, not resolved here):
    - stop_loss/take_profit: Position requires these as entry-time
      facts, but no module computes them — risk_manager.py was
      explicitly scoped to exclude them. create_position() accepts
      them as given parameters rather than computing them.
    - "update": Position's schema has no field that legitimately
      changes while a position is open (no live PnL, no trailing-stop
      level) — there is nothing to update yet. No update_position()
      function is implemented; adding one with nothing to update would
      be a placeholder, which is explicitly disallowed. This will be
      revisited if/when a field is introduced that changes post-entry.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from models.execution import OrderResult
from models.trading import Position

logger = logging.getLogger(__name__)


class PositionManagerError(Exception):
    """Raised when a position operation is given inconsistent inputs."""


def create_position(
    order_result: OrderResult,
    stop_loss: float,
    take_profit: float,
    allocated_balance: float,
    remaining_balance: float,
    artifact_id: UUID,
    entry_bar: int,
) -> Position:
    """
    Constructs a Position from a successful order execution. Called
    only after decision_engine.py has decided ENTRY and execution.py
    has filled the entry order.

    Args:
        order_result: the filled buy order (models/execution.py).
        stop_loss / take_profit: risk parameters for this position,
            supplied by the caller — not computed here.
        allocated_balance: capital committed to this position.
        remaining_balance: Trading Agent budget remaining after allocation.
        artifact_id: the PredictionArtifact the entry decision was made
            from — minimal context for decision_engine.py to re-locate
            the originating Opportunity later, without storing it here.
        entry_bar: the Forecast Cursor value at entry.

    Returns:
        Position
    """
    position = Position(
        symbol=order_result.symbol,
        entry_price=order_result.average_price,
        entry_time=datetime.utcnow(),
        amount=order_result.filled_amount,
        stop_loss=stop_loss,
        take_profit=take_profit,
        order_id=order_result.order_id,
        fee=order_result.fee_cost,
        allocated_balance=allocated_balance,
        remaining_balance=remaining_balance,
        artifact_id=artifact_id,
        entry_bar=entry_bar,
    )
    logger.info("Position created: %s @ %.8f (amount=%.8f)", position.symbol, position.entry_price, position.amount)
    return position


def close_position(position: Position, exit_order_result: OrderResult) -> None:
    """
    Closes the given Position. Called only after decision_engine.py has
    decided EXIT and execution.py has filled the exit order.

    Returns None deliberately, not a TradeRecord or any derived value:
    the caller (trading_bot.py) already holds both `position` (pre-close)
    and `exit_order_result` — everything trade_journal.py needs
    (entry_price, entry_time, amount, fee from position; average_price,
    fee_cost, filled_amount from exit_order_result) to build a
    TradeRecord is already in the caller's hands, matching
    TradeRecord's own documented contract ("produced by trade_journal.py
    from a closed Position plus its exit OrderResult/TradeFill").
    Returning a derived value here would duplicate data the caller
    already has, or couple this module to trade_journal.py's output
    shape — neither improves anything downstream. This function's real
    job is validating the pair is consistent (raising if not) and
    signaling closure; absence of an exception is the success signal.

    Raises:
        PositionManagerError: if exit_order_result does not correspond
        to this position (symbol mismatch or not a sell).
    """
    if exit_order_result.symbol != position.symbol:
        raise PositionManagerError(
            f"Exit order symbol {exit_order_result.symbol!r} does not match "
            f"open position symbol {position.symbol!r}"
        )
    if exit_order_result.side != "sell":
        raise PositionManagerError(
            f"Exit order for {position.symbol} has side={exit_order_result.side!r}, expected 'sell'"
        )

    logger.info(
        "Position closed: %s @ %.8f (entry was %.8f)",
        position.symbol, exit_order_result.average_price, position.entry_price,
    )
    return None


def is_position_open(position: Optional[Position]) -> bool:
    """
    Single-active-position query: True if a Position is currently open.
    """
    return position is not None