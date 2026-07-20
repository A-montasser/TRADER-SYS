"""
prediction_agent/scanner.py

Market Scanning module for the Prediction Agent.

Responsibility (frozen):
    Enumerate all currently active spot markets on the configured exchange.
    Produces a raw, unranked, unfiltered universe of candidate symbols.

Explicitly NOT this module's responsibility:
    - Halal keyword filtering            -> prediction_agent/filters.py
    - Quote-currency policy filtering     -> prediction_agent/filters.py
    - Liquidity / min-notional filtering  -> prediction_agent/filters.py
    - Symbol ranking / Top-N selection    -> prediction_agent/ranking.py
    - OHLCV download                      -> prediction_agent/downloader.py
    - Kronos inference                    -> prediction_agent/kronos_wrapper.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from exchange import Exchange
from models.market import ExchangeMarket, MarketCandidate

logger = logging.getLogger(__name__)


class ScannerError(Exception):
    """Raised when market enumeration cannot be completed."""


def scan_markets(
    exchange: Exchange,
    config: Dict[str, Any],
) -> List[MarketCandidate]:
    """
    Enumerate all currently active spot markets on the configured exchange.

    Args:
        exchange: instance of exchange.Exchange. Returns typed
                  ExchangeMarket objects — no ccxt shape is handled here.
        config: relevant slice of system.yaml / prediction_agent.yaml.
                Currently unused (no scanner-specific parameters exist yet)
                but accepted to keep the public interface stable if
                scan-level config (e.g. cache TTL) is introduced later.

    Returns:
        List[MarketCandidate]: active spot markets only, deduplicated by
        symbol, unfiltered by business policy and unranked.

    Raises:
        ScannerError: if enumeration fails for any reason (connectivity,
        exchange downtime, malformed response). Retry policy belongs to
        the caller (runtime.py), not to this function.
    """
    try:
        raw_markets: Dict[str, ExchangeMarket] = exchange.load_markets(reload=True)
    except Exception as exc:
        raise ScannerError(f"Failed to load markets from exchange: {exc}") from exc

    if not isinstance(raw_markets, dict):
        raise ScannerError(
            f"Unexpected market list type from exchange: "
            f"{type(raw_markets).__name__} (expected dict)"
        )

    candidates: List[MarketCandidate] = []
    seen_symbols = set()

    for symbol, market in raw_markets.items():
        if not isinstance(market, ExchangeMarket):
            logger.warning(
                "Skipping malformed market entry for %s: not an ExchangeMarket", symbol
            )
            continue

        if symbol in seen_symbols:
            logger.warning("Duplicate market entry for %s — skipping duplicate", symbol)
            continue

        if not market.spot:
            continue

        if not market.active:
            continue

        if not market.base or not market.quote:
            logger.warning(
                "Skipping %s — missing base/quote in market metadata", symbol
            )
            continue

        candidates.append(
            MarketCandidate(
                symbol=market.symbol,
                base=market.base,
                quote=market.quote,
                market_type="spot",
                active=True,
            )
        )
        seen_symbols.add(symbol)

    logger.info(
        "Scanner: %d active spot markets found (of %d total market entries)",
        len(candidates),
        len(raw_markets),
    )

    return candidates