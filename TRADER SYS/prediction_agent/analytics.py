"""
prediction_agent/analytics.py

Forecast Analytics module for the Prediction Agent.

Responsibility (frozen):
    Internal utility only. Computes objective, descriptive statistics
    from a ForecastResult (return, range, drawdown, upside). Does not
    cross the Prediction Agent -> Trading Agent boundary — the frozen
    PredictionRecord/PredictionArtifact contract does not carry these
    values (see architecture review: every field here is a deterministic
    function of ForecastSeries, which the Trading Agent already receives
    in full; scenario_analyzer.py derives equivalents itself).

Explicitly NOT this module's responsibility:
    - Buy/sell/hold recommendations   -> trading/decision_engine.py
    - Opportunity/risk scoring         -> trading/opportunity_ranker.py, risk_manager.py
    - Position sizing                  -> trading/capital_manager.py
    - Prediction Artifact construction -> prediction_agent/artifact_builder.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from models.artifact import ForecastResult

logger = logging.getLogger(__name__)


class AnalyticsError(Exception):
    """Raised when analytics cannot be computed for a forecast."""


@dataclass(frozen=True)
class ForecastAnalytics:
    """
    Module-local — never crosses the Prediction Agent -> Trading Agent
    boundary. Used only for internal diagnostics/logging.
    """
    symbol: str
    forecast_return_pct: float
    max_predicted_high: float
    min_predicted_low: float
    expected_range_pct: float
    drawdown_estimate_pct: float
    upside_estimate_pct: float


def _max_drawdown_pct(closes: List[float]) -> float:
    peak = closes[0]
    max_dd = 0.0
    for price in closes:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak * 100
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def _max_upside_pct(closes: List[float]) -> float:
    trough = closes[0]
    max_up = 0.0
    for price in closes:
        if price < trough:
            trough = price
        upside = (price - trough) / trough * 100
        if upside > max_up:
            max_up = upside
    return max_up


def compute_analytics(forecast_result: ForecastResult) -> ForecastAnalytics:
    """
    Computes descriptive analytics for one symbol's forecast.

    Raises:
        AnalyticsError: if the forecast has fewer than 2 bars.
    """
    bars = forecast_result.forecast.bars
    if len(bars) < 2:
        raise AnalyticsError(
            f"Forecast for {forecast_result.symbol} has {len(bars)} bar(s); "
            "at least 2 required to compute analytics"
        )

    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]

    first_close = closes[0]
    last_close = closes[-1]

    forecast_return_pct = (last_close - first_close) / first_close * 100
    max_predicted_high = max(highs)
    min_predicted_low = min(lows)
    expected_range_pct = (max_predicted_high - min_predicted_low) / first_close * 100

    return ForecastAnalytics(
        symbol=forecast_result.symbol,
        forecast_return_pct=forecast_return_pct,
        max_predicted_high=max_predicted_high,
        min_predicted_low=min_predicted_low,
        expected_range_pct=expected_range_pct,
        drawdown_estimate_pct=_max_drawdown_pct(closes),
        upside_estimate_pct=_max_upside_pct(closes),
    )


def compute_analytics_batch(
    forecast_results: tuple[ForecastResult, ...],
) -> tuple[ForecastAnalytics, ...]:
    """
    Computes analytics for each ForecastResult. A single symbol's
    failure is logged and skipped — does not abort the batch.
    """
    results: List[ForecastAnalytics] = []
    for forecast_result in forecast_results:
        try:
            results.append(compute_analytics(forecast_result))
        except AnalyticsError as exc:
            logger.warning("Skipping analytics for %s: %s", forecast_result.symbol, exc)
            continue

    logger.info(
        "Analytics: %d/%d forecasts processed", len(results), len(forecast_results)
    )
    return tuple(results)