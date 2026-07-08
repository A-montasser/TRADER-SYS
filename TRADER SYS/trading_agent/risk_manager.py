"""
trading_agent/risk_manager.py

Risk Evaluation module for the Trading Agent.

Responsibility (frozen):
    Compute cost-adjusted and forecast-path-derived risk metrics for a
    single Opportunity, producing a RiskAssessment (models/trading.py).
    Pure calculation only — never decides whether to trade.

Explicitly NOT this module's responsibility:
    - Final trade / no-trade decision -> trading_agent/decision_engine.py
    - Opportunity qualification/order   -> trading_agent/oppurtunity_ranker.py
    - Position sizing / allocation       -> trading_agent/capital_manager.py
    - Execution                           -> trading_agent/execution.py
"""

from __future__ import annotations

from typing import Optional

from models.trading import Opportunity, RiskAssessment

# Temporary default fee rate until exchange.yaml / live exchange fee
# lookup exists. Round-trip (entry + exit) fee is 2x this rate.
# Replace once exchange configuration is implemented.
DEFAULT_BINANCE_FEE_RATE = 0.001


def _estimated_fee_pct() -> float:
    """Round-trip fee estimate, expressed as a percent to match forecast_return_pct's units."""
    return 2 * DEFAULT_BINANCE_FEE_RATE * 100


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
    )