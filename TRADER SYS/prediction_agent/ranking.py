"""
prediction_agent/ranking.py

Symbol Ranking module for the Prediction Agent.

Responsibility (frozen):
    Ranking Framework (stable):
        - Collect ranking features per candidate.
        - Compute a sortable ranking key from those features.
        - Sort candidates by that key.
        - Truncate to the Top-N symbols that proceed to OHLCV download
          and Kronos inference.

    Current Ranking Strategy (implemented today, not the definition of
    ranking itself):
        - Single feature: momentum factor, ported verbatim from
          trading_bot.py's check_live_momentum() — the only Kronos-
          independent, trading-history-independent signal available to
          the Prediction Agent before Kronos inference occurs.
        - Tie-break: quote_volume_24h, when metrics are supplied.

    Future ranking features (liquidity, spread, volatility, ATR,
    exchange constraints, etc.) plug into _compute_ranking_features()
    and _ranking_key() without changing the framework in rank_markets().

Explicitly NOT this module's responsibility:
    - OHLCV download        -> prediction_agent/downloader.py
    - Data validation        -> prediction_agent/validator.py
    - Kronos inference       -> prediction_agent/kronos_wrapper.py
    - Trading decisions      -> trading/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional

from exchange import Exchange, ExchangeError
from models.market import MarketCandidate, MarketMetrics, RankedCandidate

logger = logging.getLogger(__name__)


class RankingError(Exception):
    """Raised when ranking cannot be completed due to invalid configuration."""


@dataclass(frozen=True)
class RankingConfig:
    """
    Mirrors prediction_agent.yaml ranking parameters.
    Owned by ranking.py — single consumer, constructed by runtime.py
    from parsed YAML.
    """
    top_n: int
    momentum_timeframe: str = "1m"
    momentum_limit: int = 20
    momentum_threshold_pct: float = 0.005
    momentum_up_factor: float = 1.2
    momentum_down_factor: float = 0.8
    momentum_neutral_factor: float = 1.0


class _RankingFeatures(NamedTuple):
    """
    Internal, module-local accumulator of per-candidate ranking features.
    Current Ranking Strategy populates only momentum_factor + volume.
    Extend this tuple (and _compute_ranking_features / _ranking_key)
    when a new feature is added — this is the intended extension seam,
    not a redesign of the framework in rank_markets().
    """
    momentum_factor: float
    quote_volume_24h: float


def _compute_momentum_factor(
    exchange: Exchange,
    symbol: str,
    config: RankingConfig,
) -> float:
    """
    Current Ranking Strategy — momentum feature.
    Repository logic ported from trading_bot.py's check_live_momentum().
    Fail-soft: any fetch/compute failure returns the neutral factor,
    matching the original's behavior exactly.
    """
    try:
        bars = exchange.fetch_ohlcv(
            symbol, config.momentum_timeframe, limit=config.momentum_limit
        )
    except ExchangeError as exc:
        logger.warning("Momentum check failed for %s: %s", symbol, exc)
        return config.momentum_neutral_factor

    if len(bars) < 10:
        return config.momentum_neutral_factor

    first_close = bars[0].close
    last_close = bars[-1].close
    if first_close == 0:
        return config.momentum_neutral_factor

    momentum = (last_close - first_close) / first_close

    if momentum > config.momentum_threshold_pct:
        return config.momentum_up_factor
    if momentum < -config.momentum_threshold_pct:
        return config.momentum_down_factor
    return config.momentum_neutral_factor


def _compute_ranking_features(
    candidate: MarketCandidate,
    exchange: Exchange,
    config: RankingConfig,
    metrics: Optional[Dict[str, MarketMetrics]],
) -> _RankingFeatures:
    """
    Current Ranking Strategy's feature collection step. Adding a new
    feature (liquidity, spread, volatility, ATR, ...) means adding it
    here and to _RankingFeatures — rank_markets() itself does not change.
    """
    momentum_factor = _compute_momentum_factor(exchange, candidate.symbol, config)

    quote_volume_24h = 0.0
    if metrics is not None:
        m = metrics.get(candidate.symbol)
        if m is not None and m.quote_volume_24h is not None:
            quote_volume_24h = m.quote_volume_24h

    return _RankingFeatures(
        momentum_factor=momentum_factor,
        quote_volume_24h=quote_volume_24h,
    )


def _ranking_key(features: _RankingFeatures) -> tuple:
    """
    Current Ranking Strategy's sort key: momentum factor primary,
    quote_volume_24h tie-break. Descending sort (higher = better).
    """
    return (features.momentum_factor, features.quote_volume_24h)


def rank_markets(
    candidates: List[MarketCandidate],
    exchange: Exchange,
    config: RankingConfig,
    metrics: Optional[Dict[str, MarketMetrics]] = None,
) -> List[RankedCandidate]:
    """
    Ranking Framework entry point: collects features per candidate,
    sorts by the current strategy's ranking key, truncates to top_n.

    Args:
        candidates: output of filters.py.
        exchange: instance of exchange.Exchange, used for feature lookup.
        config: ranking parameters.
        metrics: optional liquidity data, same source as filters.py's
                 metrics parameter.

    Returns:
        List[RankedCandidate]: sorted descending by ranking key,
        rank assigned 1-indexed, truncated to top_n.

    Raises:
        RankingError: if config.top_n is not a positive integer.
    """
    if config.top_n <= 0:
        raise RankingError(f"top_n must be positive, got {config.top_n}")

    if not candidates:
        logger.info("Ranking: no candidates to rank")
        return []

    scored = [
        (candidate, _compute_ranking_features(candidate, exchange, config, metrics))
        for candidate in candidates
    ]

    scored.sort(key=lambda item: _ranking_key(item[1]), reverse=True)

    top = scored[: config.top_n]

    ranked = [
        RankedCandidate(
            candidate=candidate,
            momentum_factor=features.momentum_factor,
            rank=idx + 1,
        )
        for idx, (candidate, features) in enumerate(top)
    ]

    logger.info(
        "Ranking: %d/%d candidates ranked, top_n=%d applied",
        len(ranked), len(candidates), config.top_n,
    )
    if ranked:
        logger.info(
            "Top candidate: %s (momentum_factor=%.2f, rank=%d)",
            ranked[0].candidate.symbol, ranked[0].momentum_factor, ranked[0].rank,
        )

    return ranked