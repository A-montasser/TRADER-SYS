"""
trading_agent/oppurtunity_ranker.py

Opportunity Qualification module for the Trading Agent.

Responsibility (frozen):
    Filter the Opportunity list produced by scenario_analyzer.py to
    those that pass permanent, runtime-independent qualification
    (currently: bullish forecast), preserving the Prediction Agent's
    ranking_position order unchanged. Never computes a new ranking
    score and never reorders opportunities independently of
    ranking_position — the Prediction Agent is the sole owner of
    market ranking.

    NOT timeline-aware: this module has no Forecast Cursor input and
    performs no entry-window/timing evaluation. All timeline-relative
    decisions belong exclusively to decision_engine.py.

Explicitly NOT this module's responsibility:
    - Market ranking              -> prediction_agent/ranking.py (Prediction Agent, authoritative)
    - Scenario analysis            -> trading_agent/scenario_analyzer.py
    - Timeline / entry / exit / hold decisions -> trading_agent/decision_engine.py
    - Risk / position sizing        -> trading_agent/risk_manager.py, capital_manager.py
"""

from __future__ import annotations

import logging

from models.trading import Opportunity

logger = logging.getLogger(__name__)


def _is_qualified(opportunity: Opportunity) -> bool:
    """
    Permanent, runtime-independent qualification rule: only bullish
    forecasts are considered.

    Precedent: trading_bot.py's enter_trade() directional gate (v1.2) —
    "never buy against the model's own direction" — necessary for a
    spot-only, no-shorting system. forecast_return_pct is signed, so
    this check is direct and unambiguous, and is a static property of
    the whole path (Opportunity is immutable), so it never changes for
    the artifact's lifetime.
    """
    return opportunity.forecast_return_pct > 0


def rank_opportunities(opportunities: tuple[Opportunity, ...]) -> tuple[Opportunity, ...]:
    """
    Filters opportunities to those that pass permanent qualification,
    preserving the Prediction Agent's ranking_position order.

    Does not compute a new ranking score. Ordering is asserted by
    ranking_position (the Prediction Agent's authoritative output),
    not invented here.

    Args:
        opportunities: output of scenario_analyzer.py.

    Returns:
        tuple[Opportunity, ...]: qualified opportunities, ordered
        ascending by ranking_position.
    """
    if not opportunities:
        logger.info("Opportunity ranker: no opportunities to evaluate")
        return tuple()

    ordered = sorted(opportunities, key=lambda o: o.ranking_position)
    qualified = tuple(o for o in ordered if _is_qualified(o))

    logger.info(
        "Opportunity ranker: %d/%d opportunities qualified",
        len(qualified), len(opportunities),
    )
    return qualified