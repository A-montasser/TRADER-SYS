"""
trading_agent/capital_manager.py

Capital Allocation module for the Trading Agent.

Responsibility (frozen):
    Determine how much of the Trading Agent's internal budget to
    allocate to an already-approved trade. Pure allocation only — no
    cost anticipation. Fees are realized by execution.py once an order
    actually fills; the internal budget itself is updated by
    position_manager.py. Reserving costs here would risk double-
    counting them against those later, authoritative deductions.

Explicitly NOT this module's responsibility:
    - Trade / no-trade decision   -> trading_agent/decision_engine.py
    - Prediction quality evaluation -> trading_agent/scenario_analyzer.py
    - Risk evaluation               -> trading_agent/risk_manager.py
    - Position management            -> trading_agent/position_manager.py
    - Execution / fee realization     -> trading_agent/execution.py
"""

from __future__ import annotations

from typing import Optional

from models.trading import Position, TradeDecision


def allocate_capital(
    decision: TradeDecision,
    available_budget: float,
    open_position: Optional[Position] = None,
) -> float:
    """
    Computes the capital (in USDT) to allocate to decision.opportunity.

    Single-active-position system: if a position is already open, no
    capital is allocated to a new trade. Otherwise the full available
    budget is allocated — no fee/slippage reservation; those costs are
    realized by execution.py, and the budget itself is updated by
    position_manager.py after entry.

    Args:
        decision: output of decision_engine.decide_trade(). If
            should_trade is False, no capital is allocated.
        available_budget: current Trading Agent internal budget (USDT).
        open_position: the currently open Position, if any.

    Returns:
        float: capital to allocate, in USDT. 0.0 if no trade should be
        made, a position is already open, or the budget is non-positive.
    """
    if not decision.should_trade:
        return 0.0

    if open_position is not None:
        return 0.0

    if available_budget <= 0:
        return 0.0

    return available_budget