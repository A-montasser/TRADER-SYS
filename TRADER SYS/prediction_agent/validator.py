"""
prediction_agent/validator.py

Validation module for the Prediction Agent.

Responsibility (frozen):
    Validate the OHLCV series in each DownloadedSeries for structural
    data quality (length, NaNs, chronological order, optional gap
    detection) before Kronos inference. Successfully validated series
    are wrapped as ValidatedSeries — a type-level guarantee consumed
    by kronos_wrapper.py.

Explicitly NOT this module's responsibility:
    - Kronos inference       -> prediction_agent/kronos_wrapper.py
    - Ranking modification   -> prediction_agent/ranking.py
    - Trading decisions      -> trading/
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

from models.market import DownloadedSeries, OHLCVBar, ValidatedSeries

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidatorConfig:
    """
    Mirrors prediction_agent.yaml validation parameters.
    Owned by validator.py — single consumer, constructed by runtime.py
    from parsed YAML.
    """
    min_bars: int = 10
    expected_interval_ms: Optional[int] = None
    max_gap_multiplier: Optional[float] = None


def _has_valid_length(bars: List[OHLCVBar], config: ValidatorConfig) -> bool:
    return len(bars) >= config.min_bars


def _has_no_nans(bars: List[OHLCVBar]) -> bool:
    for bar in bars:
        values = (bar.open, bar.high, bar.low, bar.close, bar.volume)
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values):
            return False
    return True


def _is_chronological(bars: List[OHLCVBar]) -> bool:
    for i in range(1, len(bars)):
        if bars[i].timestamp_ms <= bars[i - 1].timestamp_ms:
            return False
    return True


def _has_no_excessive_gaps(bars: List[OHLCVBar], config: ValidatorConfig) -> bool:
    if config.expected_interval_ms is None or config.max_gap_multiplier is None:
        return True

    max_allowed = config.expected_interval_ms * config.max_gap_multiplier
    for i in range(1, len(bars)):
        delta = bars[i].timestamp_ms - bars[i - 1].timestamp_ms
        if delta > max_allowed:
            return False
    return True


def validate_series(
    downloaded_series: List[DownloadedSeries],
    config: ValidatorConfig,
) -> List[ValidatedSeries]:
    """
    Validates each DownloadedSeries's OHLCV bars for structural quality.

    Args:
        downloaded_series: output of downloader.py.
        config: validation thresholds.

    Returns:
        List[ValidatedSeries]: series that passed all checks, wrapped to
        carry the validated guarantee. Rejected series are logged and
        dropped — retry/re-download policy belongs to the caller
        (runtime.py), not to this function.
    """
    if not downloaded_series:
        logger.info("Validator: no series to validate")
        return []

    valid: List[ValidatedSeries] = []

    for series in downloaded_series:
        symbol = series.ranked_candidate.candidate.symbol
        bars = series.bars

        if not _has_valid_length(bars, config):
            logger.warning(
                "Rejected %s: %d bars < min_bars=%d",
                symbol, len(bars), config.min_bars,
            )
            continue

        if not _has_no_nans(bars):
            logger.warning("Rejected %s: NaN/None value in OHLCV data", symbol)
            continue

        if not _is_chronological(bars):
            logger.warning("Rejected %s: bars not strictly chronological", symbol)
            continue

        if not _has_no_excessive_gaps(bars, config):
            logger.warning("Rejected %s: gap exceeds max_gap_multiplier", symbol)
            continue

        valid.append(ValidatedSeries(downloaded_series=series))

    logger.info(
        "Validator: %d/%d series passed validation",
        len(valid), len(downloaded_series),
    )

    return valid