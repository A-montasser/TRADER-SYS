"""
trading_agent/decision_engine.py

The only module responsible for timeline-aware trading decisions.

Responsibility (frozen):
    Given all Prediction-Agent-ordered, permanently-qualified
    opportunities (oppurtunity_ranker.py's output, NOT cursor-filtered),
    the Forecast Cursor, current Position state, precomputed
    RiskAssessments, and (when a position is open) a LiveRiskAssessment,
    decide one of:

        ENTRY — open a new position for the given Opportunity
        WAIT  — no position open, no eligible Opportunity approved yet
        HOLD  — position open, no exit condition met
        EXIT  — position open, exit condition met

    Hybrid decision philosophy: the forecast timeline (entry_window /
    profit_window) is the PRIMARY exit mechanism. Two secondary
    override mechanisms sit alongside it, checked first since they are
    overrides, not the normal path:

        1. Emergency forecast deviation — if the OBSERVED market has
           diverged from the PREDICTED trajectory (at the current
           cursor) by more than this forecast's own worst-case
           (drawdown_estimate_pct) or best-case (upside_estimate_pct)
           movement, exit regardless of what the timeline would
           otherwise say. This is deliberately NOT "is the position
           currently profitable" — it is "is reality still behaving
           like the forecast said it would."
        2. Hold-time exceeded — if more bars have elapsed since entry
           than risk_manager.py's bars_to_profitability predicted, the
           forecast scenario failed to materialize within its own
           expected schedule.

    All timeline logic lives here and only here. Risk Manager computes
    metrics only — this module receives its output as data
    (risk_assessments, live_risk), it does not call risk_manager
    itself.

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
    - Execution / exchange access             -> trading_agent/execution.py, trading_agent/runtime.py
    - Forecast Cursor ownership               -> trading_bot.py (runtime state; passed in here, never stored)
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional
from uuid import UUID

from models.trading import (
    DecisionAction,
    LiveRiskAssessment,
    Opportunity,
    Position,
    RiskAssessment,
    TradeDecision,
)

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
    risk_assessments: Mapping[str, RiskAssessment],
    live_risk: Optional[LiveRiskAssessment],
) -> TradeDecision:
    held_opportunity = _find_held_opportunity(opportunities, open_position, current_artifact_id)

    # 1. Emergency override: forecast deviation. Checked first because
    # it is an override of the primary (timeline-driven) strategy, not
    # a replacement for it.
    if held_opportunity is not None and live_risk is not None:
        deviation = live_risk.forecast_deviation_pct
        if deviation <= -held_opportunity.drawdown_estimate_pct:
            logger.info(
                "Decision: EXIT %s (emergency — forecast_deviation_pct=%.4f%% <= -drawdown_estimate_pct=%.4f%%)",
                held_opportunity.symbol, deviation, held_opportunity.drawdown_estimate_pct,
            )
            return TradeDecision(
                action=DecisionAction.EXIT,
                should_trade=False,
                opportunity=held_opportunity,
                reason=(
                    f"emergency exit — observed price deviated {deviation:.4f}% below forecast, "
                    f"beyond this forecast's own worst-case ({held_opportunity.drawdown_estimate_pct:.4f}%)"
                ),
            )
        if deviation >= held_opportunity.upside_estimate_pct:
            logger.info(
                "Decision: EXIT %s (emergency — forecast_deviation_pct=%.4f%% >= upside_estimate_pct=%.4f%%)",
                held_opportunity.symbol, deviation, held_opportunity.upside_estimate_pct,
            )
            return TradeDecision(
                action=DecisionAction.EXIT,
                should_trade=False,
                opportunity=held_opportunity,
                reason=(
                    f"emergency exit — observed price deviated {deviation:.4f}% above forecast, "
                    f"beyond this forecast's own best-case ({held_opportunity.upside_estimate_pct:.4f}%); "
                    "exiting defensively — live price is behaving outside the range this forecast "
                    "predicted (this deviation is measured against the forecast's predicted price, "
                    "not entry price, and does not by itself imply the position is profitable)"
                ),
            )

    # 2. Primary: forecast-timeline-driven exit.
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

    # 3. Hold-time exceeded: the forecast scenario failed to
    # materialize within its own expected schedule.
    if held_opportunity is not None:
        assessment = risk_assessments.get(held_opportunity.symbol)
        if assessment is not None and assessment.bars_to_profitability is not None:
            elapsed_bars = cursor - open_position.entry_bar
            if elapsed_bars > assessment.bars_to_profitability:
                logger.info(
                    "Decision: EXIT %s (hold-time exceeded — elapsed_bars=%d > bars_to_profitability=%d)",
                    held_opportunity.symbol, elapsed_bars, assessment.bars_to_profitability,
                )
                return TradeDecision(
                    action=DecisionAction.EXIT,
                    should_trade=False,
                    opportunity=held_opportunity,
                    reason=(
                        f"hold-time exceeded — {elapsed_bars} bars elapsed since entry, "
                        f"forecast expected profitability within {assessment.bars_to_profitability}"
                    ),
                )

    # 4. Safety backstop: forecast horizon exhausted.
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
    live_risk: Optional[LiveRiskAssessment] = None,
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
            safety-state input (forecast horizon exhaustion).
        live_risk: LiveRiskAssessment for the currently held Opportunity,
            if a position is open and live price was available this
            cycle — computed by risk_manager.assess_live_risk() and
            supplied by the caller. None degrades gracefully to
            timeline-only evaluation (no emergency override check).

    Returns:
        TradeDecision with action in {ENTRY, WAIT, HOLD, EXIT}.
    """
    if open_position is not None:
        return _decide_exit_or_hold(
            opportunities, cursor, horizon_bars, open_position, current_artifact_id,
            risk_assessments, live_risk,
        )

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