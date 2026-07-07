"""
prediction_agent/downloader.py

Market Data Download module for the Prediction Agent.

Responsibility (frozen):
    Download OHLCV history for each RankedCandidate produced by
    ranking.py, producing per-symbol time-series data for validator.py.

Explicitly NOT this module's responsibility:
    - Data validation (gaps, NaNs, min-length) -> prediction_agent/validator.py
    - Kronos inference                          -> prediction_agent/kronos_wrapper.py
    - Trading decisions                         -> trading/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from exchange import Exchange, ExchangeError
from models.market import DownloadedSeries, RankedCandidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloaderConfig:
    """
    Mirrors prediction_agent.yaml download parameters.
    Owned by downloader.py — single consumer, constructed by runtime.py
    from parsed YAML.
    """
    timeframe: str
    limit: int


def download_market_data(
    ranked_candidates: List[RankedCandidate],
    exchange: Exchange,
    config: DownloaderConfig,
) -> List[DownloadedSeries]:
    """
    Downloads OHLCV history for each ranked candidate.

    Args:
        ranked_candidates: output of ranking.py.
        exchange: instance of exchange.Exchange.
        config: download parameters (timeframe, limit).

    Returns:
        List[DownloadedSeries]: one entry per candidate that returned
        non-empty data. Candidates whose download fails or returns no
        bars are dropped and logged — retry policy belongs to the
        caller (runtime.py), not to this function.
    """
    if not ranked_candidates:
        logger.info("Downloader: no candidates to download")
        return []

    results: List[DownloadedSeries] = []

    for ranked in ranked_candidates:
        symbol = ranked.candidate.symbol
        try:
            bars = exchange.fetch_ohlcv(symbol, config.timeframe, limit=config.limit)
        except ExchangeError as exc:
            logger.warning("Download failed for %s: %s", symbol, exc)
            continue

        if not bars:
            logger.warning("Download returned no bars for %s — skipping", symbol)
            continue

        results.append(DownloadedSeries(ranked_candidate=ranked, bars=bars))

    logger.info(
        "Downloader: %d/%d candidates downloaded successfully",
        len(results), len(ranked_candidates),
    )

    return results