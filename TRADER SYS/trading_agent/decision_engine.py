"""
trading_agent/decision_engine.py

The only module responsible for timeline-aware trading decisions.

Responsibility (frozen):
    Given all Prediction-Agent-ordered, permanently-qualified
    opportunities (oppurtunity_ranker.py's output, NOT cursor-filtered),
    the Forecast Cursor, current Position state, precomputed
    RiskAssessments, and minimal safety state, decide one of:

        ENTRY — open a new position for the given Opportunity
        WAIT  — no position open, no eligible Opportunity approved yet
        HOLD  — position open, no exit condition met
        EXIT  — position open, exit condition met

    All timeline logic (entry-window eligibility, profit-window exit,
    horizon-exhaustion safety exit) lives here and only here. Risk
    Manager computes metrics only — this module receives its output as
    data (risk_assessments), it does not call risk_manager itself, to
    keep this module the single place trading policy is decided rather
    than also owning an internal orchestration dependency on Risk
    Manager.

    Does not track a separately-held Opportunity as runtime state: when
    a Position is open, the Opportunity that justified it is re-located
    by matching Position.symbol against the same `opportunities` list
    passed in this cycle (safe because Opportunity is immutable and
    static for the artifact's lifetime), guarded by Position.artifact_id
    matching the current artifact.

Explicitly NOT this module's responsibility:
    - Market ranking                    -> prediction_agent/ranking.py
    - Opportunity qualification/order    -> trading_agent/oppurtunity_ranker.py
    - Risk metric computation             -> trading_agent/risk_manager.py
    - Position sizing / allocation         -> trading_agent/capital_manager.py
    - Position lifecycle (create/close)     -> trading_agent/position_manager.py
    - Execution                             -> trading_agent/execution.py
    - Forecast Cursor ownership               -> trading_bot.py (runtime state; passed in here, never stored)
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional
from uuid import UUID

from models.trading import DecisionAction, Opportunity, Position, RiskAssessment, TradeDecision

logger = logging.getLogger(__name__)


def _find_held_opportunity(
    opportunities: tuple[Opportunity, ...],
    open_position: Position,
    current_artifact_id: UUID,
) -> Optional[Opportunity]:
    """
    Re-locates the Opportunity behind open_position from the given
    opportunities list, without any separately tracked runtime state.
    Returns None if the position belongs to a different (stale)
    artifact, or if its symbol is no longer present in the list — both
    treated identically by the caller as "no timeline reference
    available," falling back to safety-only evaluation.
    """
    if open_position.artifact_id != current_artifact_id:
        return None
    return next((o for o in opportunities if o.symbol == open_position.symbol), None)


def _decide_exit_or_hold(
    opportunities: tuple[Opportunity, ...],
    cursor: int,
    horizon_bars: int,
    open_position: Position,
    current_artifact_id: UUID,
) -> TradeDecision:
    held_opportunity = _find_held_opportunity(opportunities, open_position, current_artifact_id)

    if held_opportunity is not None and cursor >= held_opportunity.profit_window.end_bar:
        logger.info(
            "Decision: EXIT %s (cursor=%d reached profit_window.end_bar=%d)",
            held_opportunity.symbol, cursor, held_opportunity.profit_window.end_bar,
        )
        return TradeDecision(
            action=DecisionAction.EXIT,
            should_trade=False,
            opportunity=held_opportunity,
            reason=f"cursor={cursor} reached profit_window end (bar {held_opportunity.profit_window.end_bar})",
        )

    if cursor >= horizon_bars - 1:
        logger.info("Decision: EXIT (safety — forecast horizon exhausted at cursor=%d)", cursor)
        return TradeDecision(
            action=DecisionAction.EXIT,
            should_trade=False,
            opportunity=held_opportunity,
            reason=f"safety exit — cursor={cursor} reached forecast horizon ({horizon_bars} bars)",
        )

    logger.debug("Decision: HOLD (cursor=%d, no exit condition met)", cursor)
    return TradeDecision(
        action=DecisionAction.HOLD,
        should_trade=False,
        opportunity=held_opportunity,
        reason="position open, no exit condition met",
    )


def decide(
    opportunities: tuple[Opportunity, ...],
    cursor: int,
    open_position: Optional[Position],
    risk_assessments: Mapping[str, RiskAssessment],
    current_artifact_id: UUID,
    horizon_bars: int,
) -> TradeDecision:
    """
    Makes the final trading decision for the current cycle.

    Args:
        opportunities: output of oppurtunity_ranker.rank_opportunities()
            — all permanently-qualified opportunities, in Prediction
            Agent order. NOT pre-filtered by cursor; entry-window
            eligibility is evaluated here.
        cursor: current Forecast Cursor position (0-indexed bar),
            owned by trading_bot.py.
        open_position: the currently open Position, if any.
        risk_assessments: precomputed RiskAssessment per candidate
            Opportunity, keyed by symbol. Computed by risk_manager.py
            and supplied by the caller — this module does not call
            risk_manager itself.
        current_artifact_id: the artifact_id opportunities was derived
            from, used to detect a stale Position (see
            _find_held_opportunity).
        horizon_bars: the artifact's forecast horizon (pred_len) — the
            minimal safety-state input implemented now (forecast
            horizon exhaustion). A fuller safety-state concept (e.g.
            daily loss limits) may be added later without redesigning
            this signature.

    Returns:
        TradeDecision with action in {ENTRY, WAIT, HOLD, EXIT}.
    """
    if open_position is not None:
        return _decide_exit_or_hold(opportunities, cursor, horizon_bars, open_position, current_artifact_id)

    for opportunity in opportunities:
        if not (opportunity.entry_window.start_bar <= cursor <= opportunity.entry_window.end_bar):
            logger.debug("Decision: %s not yet in entry_window at cursor=%d", opportunity.symbol, cursor)
            continue

        assessment = risk_assessments.get(opportunity.symbol)
        if assessment is None:
            logger.debug("Decision: no RiskAssessment supplied for %s — skipping", opportunity.symbol)
            continue

        if assessment.net_expected_profit_pct > 0:
            logger.info(
                "Decision: ENTRY %s (ranking_position=%d, cursor=%d)",
                opportunity.symbol, opportunity.ranking_position, cursor,
            )
            return TradeDecision(
                action=DecisionAction.ENTRY,
                should_trade=True,
                opportunity=opportunity,
                reason=f"approved at cursor={cursor} (net_expected_profit_pct > 0)",
            )
        logger.debug("Decision: %s rejected at cursor=%d (not net-profitable)", opportunity.symbol, cursor)

    logger.info("Decision: WAIT (cursor=%d, no eligible opportunity approved)", cursor)
    return TradeDecision(
        action=DecisionAction.WAIT,
        should_trade=False,
        opportunity=None,
        reason="no eligible opportunity approved this cycle",
    )