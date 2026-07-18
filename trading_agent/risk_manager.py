"""
trading_agent/risk_manager.py

Risk Evaluation module for the Trading Agent.

Responsibility (frozen):
    Compute cost-adjusted, forecast-path-derived, and live-price-
    derived risk metrics. Two entry points:

        assess_risk(opportunity) -> RiskAssessment
            Static — computed once per candidate Opportunity from the
            whole forecast path alone. Used for ENTRY evaluation, and
            as the source of stop_loss_pct/take_profit_pct.

        assess_live_risk(opportunity, cursor, current_price) -> LiveRiskAssessment
            Dynamic — computed fresh each cycle a position is open,
            using live price. Measures how far the OBSERVED market has
            diverged from the PREDICTED trajectory at the current
            Forecast Cursor position ("is reality still following the
            forecast", not "what is the current profit").

    Pure calculation only in both — never decides whether to trade,
    never accesses the exchange (current_price is received as a plain
    number, fetched by the caller).

Explicitly NOT this module's responsibility:
    - Final trade / no-trade decision -> trading_agent/decision_engine.py
    - Opportunity qualification/order   -> trading_agent/oppurtunity_ranker.py
    - Position sizing / allocation       -> trading_agent/capital_manager.py
    - Execution / exchange access          -> trading_agent/execution.py, trading_agent/runtime.py
"""

from __future__ import annotations

from typing import Optional

from models.trading import LiveRiskAssessment, Opportunity, RiskAssessment

# Temporary default fee rate until exchange.yaml / live exchange fee
# lookup exists. Round-trip (entry + exit) fee is 2x this rate.
# Replace once exchange configuration is implemented.
DEFAULT_BINANCE_FEE_RATE_FRACTION = 0.001


def _estimated_fee_pct() -> float:
    """Round-trip fee estimate, expressed as a percent to match forecast_return_pct's units."""
    return 2 * DEFAULT_BINANCE_FEE_RATE_FRACTION * 100


def _estimated_slippage_pct() -> float:
    """
    Temporary zero-slippage assumption. No repository information
    currently supports a realistic slippage model — this will be
    replaced by a real execution-layer estimate once available.
    """
    return 0.0


def _reward_to_risk_ratio(net_expected_profit_pct: float, drawdown_estimate_pct: float) -> float:
    if drawdown_estimate_pct > 0:
        return net_expected_profit_pct / drawdown_estimate_pct
    return float("inf") if net_expected_profit_pct > 0 else 0.0


def _bars_to_profitability(opportunity: Opportunity, total_cost_pct: float) -> Optional[int]:
    """
    Walks the forecast path and returns the number of bars (1-indexed)
    until cumulative return from the first bar's close covers
    total_cost_pct. None if the path never covers costs within horizon.
    """
    bars = opportunity.record.forecast.bars
    if not bars:
        return None

    first_close = bars[0].close
    for idx, bar in enumerate(bars):
        cumulative_return_pct = (bar.close - first_close) / first_close * 100
        if cumulative_return_pct >= total_cost_pct:
            return idx + 1
    return None


def assess_risk(opportunity: Opportunity) -> RiskAssessment:
    """
    Computes a RiskAssessment for one Opportunity. Deterministic, pure
    function — no exchange access, no decision.

    stop_loss_pct/take_profit_pct directly reuse
    opportunity.drawdown_estimate_pct/upside_estimate_pct — this
    forecast's own worst/best-case movement is the boundary, not an
    independently invented number.

    Args:
        opportunity: output of oppurtunity_ranker.rank_opportunities().

    Returns:
        RiskAssessment
    """
    estimated_fee_pct = _estimated_fee_pct()
    estimated_slippage_pct = _estimated_slippage_pct()
    total_cost_pct = estimated_fee_pct + estimated_slippage_pct

    net_expected_profit_pct = opportunity.forecast_return_pct - total_cost_pct

    return RiskAssessment(
        estimated_fee_pct=estimated_fee_pct,
        estimated_slippage_pct=estimated_slippage_pct,
        net_expected_profit_pct=net_expected_profit_pct,
        reward_to_risk_ratio=_reward_to_risk_ratio(
            net_expected_profit_pct, opportunity.drawdown_estimate_pct
        ),
        bars_to_profitability=_bars_to_profitability(opportunity, total_cost_pct),
        stop_loss_pct=opportunity.drawdown_estimate_pct,
        take_profit_pct=opportunity.upside_estimate_pct,
    )


def assess_live_risk(
    opportunity: Opportunity,
    cursor: int,
    current_price: float,
) -> Optional[LiveRiskAssessment]:
    """
    Computes how far the observed current_price has diverged from the
    forecast's predicted close at the current Forecast Cursor position.

    Args:
        opportunity: the Opportunity behind the currently open Position.
        cursor: current Forecast Cursor position (0-indexed bar).
        current_price: live market price, fetched by the caller — this
            function never accesses the exchange itself.

    Returns:
        LiveRiskAssessment, or None if cursor falls outside the
        forecast path (nothing to compare against) or the predicted
        price at that bar is zero (division not meaningful).
    """
    bars = opportunity.record.forecast.bars
    if not (0 <= cursor < len(bars)):
        return None

    predicted_close = bars[cursor].close
    if predicted_close == 0:
        return None

    forecast_deviation_pct = (current_price - predicted_close) / predicted_close * 100
    return LiveRiskAssessment(forecast_deviation_pct=forecast_deviation_pct)