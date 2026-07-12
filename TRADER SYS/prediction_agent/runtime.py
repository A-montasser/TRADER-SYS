"""
prediction_agent/runtime.py

Runtime orchestrator for the Prediction Agent.

Responsibility (frozen):
    Execute exactly one complete prediction cycle by coordinating the
    already-implemented Prediction Agent modules in order: scan,
    filter, rank, download, validate, forecast, analyze, build, and
    persist a PredictionArtifact. Pure orchestration — no new
    prediction algorithms, no trading logic.

Explicitly NOT this module's responsibility:
    - Any prediction algorithm itself -> scanner.py, filters.py, ranking.py,
      downloader.py, validator.py, kronos_wrapper.py, artifact_builder.py
    - Trading decisions                 -> trading_agent/
    - Scheduling / retry / monitoring     -> trading_bot.py
    - Looping / sleeping                    -> trading_bot.py

This module executes ONE cycle and returns. It does not loop, does not
sleep, does not retry, and does not schedule future runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from exchange import Exchange, ExchangeError
from models.artifact import PredictionArtifact
from models.market import MarketMetrics
from prediction_agent.analytics import compute_analytics_batch
from prediction_agent.artifact_builder import ArtifactBuilderConfig, build_artifact, save_artifact
from prediction_agent.downloader import DownloaderConfig, download_market_data
from prediction_agent.filters import FilterConfig, apply_filters
from prediction_agent.kronos_wrapper import KronosWrapper
from prediction_agent.ranking import RankingConfig, rank_markets
from prediction_agent.scanner import scan_markets
from prediction_agent.validator import ValidatorConfig, validate_series

logger = logging.getLogger(__name__)


def _fetch_metrics(exchange: Exchange, symbols: List[str]) -> Optional[Dict[str, MarketMetrics]]:
    """
    Best-effort metrics fetch. filters.py/ranking.py both already
    document graceful degradation when metrics is None — a fetch
    failure here degrades to that existing behavior rather than
    aborting the cycle.
    """
    if not symbols:
        return None
    try:
        return exchange.fetch_market_metrics(symbols)
    except ExchangeError as exc:
        logger.warning("Failed to fetch market metrics — continuing without them: %s", exc)
        return None


def run_prediction_cycle(
    exchange: Exchange,
    kronos_wrapper: KronosWrapper,
    scanner_config: Dict[str, Any],
    filter_config: FilterConfig,
    ranking_config: RankingConfig,
    downloader_config: DownloaderConfig,
    validator_config: ValidatorConfig,
    artifact_builder_config: ArtifactBuilderConfig,
    parquet_path: Path,
    meta_path: Path,
) -> PredictionArtifact:
    """
    Executes one complete prediction cycle and returns the resulting
    PredictionArtifact (already persisted to parquet_path/meta_path).

    Args:
        exchange: instance of exchange.Exchange.
        kronos_wrapper: already-constructed KronosWrapper (expensive to
            build — owned and reused by the caller across cycles).
        scanner_config / filter_config / ranking_config /
        downloader_config / validator_config / artifact_builder_config:
            existing per-module config objects.
        parquet_path / meta_path: destination for the persisted artifact.

    Returns:
        PredictionArtifact

    Raises:
        Whatever the underlying stage raises (ScannerError, RankingError,
        KronosWrapperError, ArtifactBuilderError, etc.) — not wrapped,
        so the caller can distinguish which stage failed.
    """
    candidates = scan_markets(exchange, scanner_config)
    logger.info("Runtime: %d candidates scanned", len(candidates))

    metrics = _fetch_metrics(exchange, [c.symbol for c in candidates])

    filtered = apply_filters(candidates, filter_config, metrics)
    logger.info("Runtime: %d candidates after filtering", len(filtered))

    ranked = rank_markets(filtered, exchange, ranking_config, metrics)
    logger.info("Runtime: %d candidates ranked (top_n applied)", len(ranked))

    downloaded = download_market_data(ranked, exchange, downloader_config)
    logger.info("Runtime: %d symbols downloaded", len(downloaded))

    validated = validate_series(downloaded, validator_config)
    logger.info("Runtime: %d symbols passed validation", len(validated))

    forecast_results = kronos_wrapper.generate_forecasts(validated)
    logger.info("Runtime: %d symbols forecasted", len(forecast_results))

    # Internal diagnostics only — never crosses into the artifact.
    analytics = compute_analytics_batch(forecast_results)
    for a in analytics:
        logger.info(
            "Runtime diagnostics: %s forecast_return=%.4f%% drawdown=%.4f%% upside=%.4f%%",
            a.symbol, a.forecast_return_pct, a.drawdown_estimate_pct, a.upside_estimate_pct,
        )

    artifact = build_artifact(forecast_results, ranked, artifact_builder_config)
    save_artifact(artifact, parquet_path, meta_path)
    logger.info("Runtime: prediction cycle complete, artifact %s persisted", artifact.artifact_id)

    return artifact