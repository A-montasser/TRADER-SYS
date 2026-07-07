"""
trading/scenario_analyzer.py

Scenario Analysis module for the Trading Agent.

Responsibility (frozen):
    Compute descriptive, objective trading-scenario statistics for each
    symbol in a PredictionArtifact, producing one Opportunity
    (models/trading.py) per symbol. Pure function of the artifact's
    forecast data — no trading policy, no exchange access, no Kronos.

Explicitly NOT this module's responsibility:
    - Artifact loading            -> trading/artifact_loader.py
    - Opportunity selection/order  -> trading/opportunity_ranker.py
    - Entry/exit decisions          -> trading/decision_engine.py
    - Risk / position sizing        -> trading/risk_manager.py, capital_manager.py
    - Kronos inference               -> never imported here
"""

from __future__ import annotations

import logging
from typing import List

from models.artifact import PredictionArtifact, PredictionRecord
from models.trading import Opportunity

logger = logging.getLogger(__name__)


class ScenarioAnalyzerError(Exception):
    """Raised when scenario analysis cannot be completed for the artifact as a whole."""


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


def _analyze_record(record: PredictionRecord) -> Opportunity:
    """
    Raises:
        ScenarioAnalyzerError: if the record has fewer than 2 bars.
    """
    bars = record.forecast.bars
    if len(bars) < 2:
        raise ScenarioAnalyzerError(
            f"Forecast for {record.symbol} has {len(bars)} bar(s); "
            "at least 2 required to compute scenario statistics"
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

    return Opportunity(
        record=record,
        symbol=record.symbol,
        ranking_position=record.ranking_position,
        ranking_score=record.ranking_score,
        forecast_return_pct=forecast_return_pct,
        max_predicted_high=max_predicted_high,
        min_predicted_low=min_predicted_low,
        expected_range_pct=expected_range_pct,
        drawdown_estimate_pct=_max_drawdown_pct(closes),
        upside_estimate_pct=_max_upside_pct(closes),
    )


def analyze_scenarios(artifact: PredictionArtifact) -> tuple[Opportunity, ...]:
    """
    Computes one Opportunity per PredictionRecord in the artifact.

    A single record's failure (fewer than 2 bars) is logged and skipped —
    it does not abort analysis of the remaining records.

    Returns:
        tuple[Opportunity, ...]: one per record that could be analyzed,
        in the same order as artifact.records.
    """
    if not artifact.records:
        logger.info("Scenario analyzer: artifact has no records")
        return tuple()

    opportunities: List[Opportunity] = []
    for record in artifact.records:
        try:
            opportunities.append(_analyze_record(record))
        except ScenarioAnalyzerError as exc:
            logger.warning("Skipping scenario analysis for %s: %s", record.symbol, exc)
            continue

    logger.info(
        "Scenario analyzer: %d/%d records analyzed",
        len(opportunities), len(artifact.records),
    )
    return tuple(opportunities)