"""
trading_agent/position_manager.py

Position Lifecycle module for the Trading Agent.

Responsibility (frozen):
    Own the lifecycle of the Trading Agent's single open Position:
    construct it from a successful order execution, and expose whether
    a position is currently open. Single-active-position system.

Explicitly NOT this module's responsibility:
    - Trade / no-trade decision       -> trading_agent/decision_engine.py
    - Prediction quality evaluation     -> trading_agent/scenario_analyzer.py
    - Risk evaluation                    -> trading_agent/risk_manager.py
    - Capital allocation                  -> trading_agent/capital_manager.py
    - Order execution                      -> trading_agent/execution.py
    - Closing / trade history                -> trading_agent/trade_journal.py

Implementation constraint (flagged, not resolved here):
    Position requires stop_loss/take_profit as entry-time facts, but no
    Stage 3 module currently computes them — risk_manager.py was
    explicitly scoped to exclude them. create_position() accepts them
    as given parameters rather than computing them, since computing a
    stop-loss/take-profit level is a risk-evaluation decision this
    module is not permitted to make.

    Similarly, "monitoring the active position" implies exit-condition
    evaluation against live price, which requires exchange access this
    module does not have (consistent with every other Stage 3 module).
    Only a structural open/closed query is implemented here; live
    exit-trigger evaluation is deferred to a module with exchange
    access.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from models.execution import OrderResult
from models.trading import Position


def create_position(
    order_result: OrderResult,
    stop_loss: float,
    take_profit: float,
    allocated_balance: float,
    remaining_balance: float,
) -> Position:
    """
    Constructs a Position from a successful order execution.

    Args:
        order_result: the filled buy order (models/execution.py).
        stop_loss / take_profit: risk parameters for this position,
            supplied by the caller — not computed here.
        allocated_balance: capital committed to this position.
        remaining_balance: Trading Agent budget remaining after allocation.

    Returns:
        Position
    """
    return Position(
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
    )


def is_position_open(position: Optional[Position]) -> bool:
    """
    Single-active-position query: True if a Position is currently open.
    """
    return position is not None