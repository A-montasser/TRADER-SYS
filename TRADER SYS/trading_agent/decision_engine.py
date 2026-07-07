"""
trading/decision_engine.py

Entry Decision module for the Trading Agent.

Responsibility (frozen):
    Orchestrate the entry decision: iterate the qualified opportunities
    produced by opportunity_ranker.py strictly in Prediction Agent
    order, present each to the Risk Manager for approval, and return
    the first approved opportunity as a trade decision. Never reorders
    or rescoring opportunities — the Prediction Agent's ranking remains
    authoritative.

    This module owns NO trading policy or numerical threshold (minimum
    return, drawdown, volatility, reward/risk, fees, slippage, capital).
    Those belong to trading/risk_manager.py. Until risk_manager.py
    exists, every opportunity presented here has already been qualified
    by opportunity_ranker.py, so the first one in order is accepted —
    no threshold is applied here in its place.

Explicitly NOT this module's responsibility:
    - Market ranking                   -> prediction_agent/ranking.py
    - Opportunity qualification/order   -> trading/opportunity_ranker.py
    - Trading policy / thresholds        -> trading/risk_manager.py
    - Position sizing / allocation       -> trading/capital_manager.py
    - Stop-loss / take-profit            -> trading/risk_manager.py
    - Exit timing / hold management       -> later modules
    - Execution                           -> trading/execution.py
"""

from __future__ import annotations

import logging

from models.trading import Opportunity, TradeDecision

logger = logging.getLogger(__name__)


def decide_trade(opportunities: tuple[Opportunity, ...]) -> TradeDecision:
    """
    Iterates opportunities strictly in the order provided
    (opportunity_ranker.py's output — Prediction Agent ranking order).
    Presents each to the Risk Manager for approval and returns the
    first one approved.

    Args:
        opportunities: output of opportunity_ranker.rank_opportunities().

    Returns:
        TradeDecision: should_trade=True with the selected opportunity,
        or should_trade=False with opportunity=None if none are approved.
    """
    for opportunity in opportunities:
        # Integration point for trading/risk_manager.py (not yet
        # implemented): the Risk Manager will decide here whether this
        # opportunity may be traded, e.g.:
        #
        #     if not risk_manager.approve(opportunity):
        #         continue
        #
        # Until risk_manager.py exists, no rejection policy is applied
        # here — the first opportunity in Prediction Agent order is
        # accepted, since opportunity_ranker.py has already qualified it.
        logger.info(
            "Decision: TRADE %s (ranking_position=%d)",
            opportunity.symbol, opportunity.ranking_position,
        )
        return TradeDecision(
            should_trade=True,
            opportunity=opportunity,
            reason="first opportunity in Prediction Agent order (risk_manager.py not yet integrated)",
        )

    logger.info("Decision: SKIP — no opportunities to evaluate")
    return TradeDecision(
        should_trade=False,
        opportunity=None,
        reason="no opportunities to evaluate",
    )