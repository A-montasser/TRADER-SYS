"""
models/market.py

Shared market data contracts that cross module and agent boundaries
within the Prediction Agent pipeline (exchange.py -> scanner.py ->
filters.py -> ranking.py -> downloader.py).

These are data-only contracts. No business logic belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ExchangeMarket:
    """
    Typed representation of one exchange market, owned and constructed
    by exchange.py. Isolates ccxt's raw dict shape from the rest of the
    repository — no other module should read ccxt-specific keys directly.

    min_notional / max_notional / amount_precision are ccxt-derived facts
    (see exchange.py._extract_min_notional / _extract_max_notional /
    _extract_amount_precision) needed by future position-sizing modules
    (risk_manager.py, execution.py). Optional with safe defaults so
    scanner.py/filters.py, which don't use them, are unaffected.
    """
    symbol: str
    base: str
    quote: str
    spot: bool
    active: bool
    min_notional: float = 5.0
    max_notional: float = float("inf")
    amount_precision: int = 6


@dataclass(frozen=True)
class MarketCandidate:
    """
    Raw, structural market record produced by scanner.py.
    NOT the Prediction Artifact — internal pipeline handoff type
    consumed by filters.py, ranking.py, and downloader.py.
    """
    symbol: str
    base: str
    quote: str
    market_type: str
    active: bool


@dataclass(frozen=True)
class MarketMetrics:
    """
    Per-symbol liquidity data, sourced from exchange.py and consumed by
    filters.py's optional eligibility filter and ranking.py's tie-break.
    """
    quote_volume_24h: Optional[float] = None
    spread_pct: Optional[float] = None


@dataclass(frozen=True)
class OHLCVBar:
    """
    One OHLCV candle, produced by exchange.py and consumed by
    prediction_agent/downloader.py, validator.py, kronos_wrapper.py,
    and ranking.py (momentum feature computation).

    Field order matches ccxt's standard OHLCV array
    ([timestamp, open, high, low, close, volume]) — confirmed by
    trading_bot.py's check_live_momentum() indexing ohlcv[i][0] and
    ohlcv[i][4].
    """
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class RankedCandidate:
    """
    MarketCandidate enriched with ranking metadata, produced by
    ranking.py. Crosses ranking.py -> downloader.py -> validator.py,
    hence a shared model rather than module-local.

    momentum_factor: the current ranking strategy's feature value
        (momentum multiplier, ported from trading_bot.py's
        check_live_momentum()). Named explicitly rather than "score"
        because it is one specific feature, not a generic composite —
        see ranking.py for the Ranking Framework / Ranking Strategy
        separation. If ranking later combines multiple features into
        a single sortable value, introduce a separate generic
        `ranking_score` field at that time rather than overloading
        this one.
    rank: 1-indexed position after sorting (1 = highest priority).
    """
    candidate: MarketCandidate
    momentum_factor: float
    rank: int

@dataclass(frozen=True)
class DownloadedSeries:
    """
    RankedCandidate paired with its downloaded OHLCV history, produced by
    downloader.py. Crosses downloader.py -> validator.py -> (Stage 3)
    kronos_wrapper.py, hence a shared model rather than module-local.
    """
    ranked_candidate: RankedCandidate
    bars: List[OHLCVBar]

@dataclass(frozen=True)
class ValidatedSeries:
    """
    DownloadedSeries that has passed all structural validation rules in
    validator.py (minimum length, chronological order, no NaNs, acceptable
    gaps when enabled). Constructed only by validator.py, after every rule
    passes. Consumed by kronos_wrapper.py (Stage 3) — no other module may
    construct this type, preserving the "validated" guarantee as a type-
    level fact rather than a runtime convention.
    """
    downloaded_series: DownloadedSeries
    @property
    def symbol(self) -> str:
        """
        Convenience accessor that hides the internal pipeline structure
        from downstream consumers.
         """
        return self.downloaded_series.ranked_candidate.candidate.symbol

    @property
    def bars(self) -> List[OHLCVBar]:
        """
        Convenience accessor that exposes validated OHLCV bars without
        requiring downstream modules to know DownloadedSeries exists.
        """
        return self.downloaded_series.bars