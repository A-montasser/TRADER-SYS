"""
prediction_agent/filters.py

Market Filtering + Halal Filtering module for the Prediction Agent.

Responsibility (frozen):
    Filter the MarketCandidate list produced by scanner.py using:
        1. Quote currency policy
        2. Halal keyword policy
        3. Basic market eligibility (liquidity / volume / spread) — config-gated,
           uses MarketMetrics supplied by the caller (sourced from exchange.py).

Explicitly NOT this module's responsibility:
    - OHLCV download        -> prediction_agent/downloader.py
    - Symbol ranking         -> prediction_agent/ranking.py
    - Kronos inference       -> prediction_agent/kronos_wrapper.py
    - Trading decisions      -> trading/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from models.market import MarketCandidate, MarketMetrics

logger = logging.getLogger(__name__)


# Repository fact: HARAM_KEYWORDS / is_haram_symbol() in trading_bot.py.
# Reused verbatim as the default list; config may override/extend it.
DEFAULT_HARAM_KEYWORDS: List[str] = [
    'gambling', 'casino', 'meme', 'pepe', 'scam',
    'pump', 'dump', 'nsfw', 'adult', 'anonymous',
    'ponzi', 'rug', 'high-risk', 'betting',
]


@dataclass(frozen=True)
class FilterConfig:
    """
    Mirrors prediction_agent.yaml filtering parameters.
    Owned by filters.py — single consumer, constructed by runtime.py
    from parsed YAML.
    """
    allowed_quote_currencies: List[str]
    haram_keywords: List[str]
    min_quote_volume_24h: Optional[float] = None
    max_spread_pct: Optional[float] = None


def is_haram_symbol(symbol: str, haram_keywords: List[str]) -> bool:
    """
    Repository logic from trading_bot.py's is_haram_symbol(), parameterized
    by an injected keyword list instead of the module-level constant.
    """
    coin = symbol.split('/')[0].lower()
    return any(keyword in coin for keyword in haram_keywords)


def filter_by_quote_currency(
    candidates: List[MarketCandidate],
    allowed_quotes: List[str],
) -> List[MarketCandidate]:
    """
    Keeps only candidates whose quote currency is in allowed_quotes.
    Comparison is case-insensitive.
    """
    allowed = {q.upper() for q in allowed_quotes}
    result = [c for c in candidates if c.quote.upper() in allowed]

    logger.info(
        "Quote currency filter: %d/%d markets kept (allowed=%s)",
        len(result), len(candidates), sorted(allowed),
    )
    return result


def filter_halal(
    candidates: List[MarketCandidate],
    haram_keywords: List[str],
) -> List[MarketCandidate]:
    """
    Removes candidates whose base asset matches any haram keyword.
    """
    result = [
        c for c in candidates
        if not is_haram_symbol(c.symbol, haram_keywords)
    ]

    rejected = len(candidates) - len(result)
    if rejected:
        logger.info("Halal filter: %d markets rejected", rejected)
    return result


def filter_by_market_metrics(
    candidates: List[MarketCandidate],
    config: FilterConfig,
    metrics: Optional[Dict[str, MarketMetrics]] = None,
) -> List[MarketCandidate]:
    """
    Applies minimum liquidity / volume and maximum spread thresholds.

    `metrics` is sourced from exchange.py and supplied by the caller
    (runtime.py). This function does not fetch or fabricate market
    metrics itself. If no thresholds are configured, all candidates
    pass through unchanged.
    """
    if config.min_quote_volume_24h is None and config.max_spread_pct is None:
        return candidates

    if metrics is None:
        logger.warning(
            "Liquidity/spread thresholds are configured but no market "
            "metrics were supplied by the caller — skipping this filter stage"
        )
        return candidates

    result: List[MarketCandidate] = []
    for candidate in candidates:
        m = metrics.get(candidate.symbol)
        if m is None:
            logger.debug(
                "No metrics available for %s — excluding (cannot verify "
                "eligibility)", candidate.symbol,
            )
            continue

        if (config.min_quote_volume_24h is not None
                and (m.quote_volume_24h is None
                     or m.quote_volume_24h < config.min_quote_volume_24h)):
            continue

        if (config.max_spread_pct is not None
                and (m.spread_pct is None
                     or m.spread_pct > config.max_spread_pct)):
            continue

        result.append(candidate)

    logger.info(
        "Market metrics filter: %d/%d markets kept",
        len(result), len(candidates),
    )
    return result


def apply_filters(
    candidates: List[MarketCandidate],
    config: FilterConfig,
    metrics: Optional[Dict[str, MarketMetrics]] = None,
) -> List[MarketCandidate]:
    """
    Runs the full filter pipeline in order:
        quote currency -> halal -> market metrics (optional)

    This is the single public entry point runtime.py should call.
    """
    result = filter_by_quote_currency(candidates, config.allowed_quote_currencies)
    result = filter_halal(result, config.haram_keywords)
    result = filter_by_market_metrics(result, config, metrics)

    logger.info(
        "Filter pipeline complete: %d/%d markets eligible",
        len(result), len(candidates),
    )
    return result