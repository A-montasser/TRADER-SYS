"""
exchange.py

Sole translation boundary between ccxt and the rest of the repository.
No other module may import ccxt or access raw exchange dictionaries.

Responsibilities:
    - Exchange connectivity (credentials, connection options)
    - Market enumeration, translated to ExchangeMarket
    - Ticker / OHLCV data retrieval
    - Order placement and order/trade history retrieval

Explicitly NOT this module's responsibility:
    - Position sizing decisions      -> trading/risk_manager.py, capital_manager.py
    - Retry / backoff orchestration  -> runtime.py, trading/execution.py
    - Trading decisions              -> trading/decision_engine.py
    - Business filtering / ranking   -> prediction_agent/filters.py, ranking.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import ccxt

from models.market import ExchangeMarket, MarketMetrics, OHLCVBar
from models.execution import OrderResult, TradeFill

logger = logging.getLogger(__name__)


class ExchangeError(Exception):
    """Raised when an exchange operation cannot be completed."""


class Exchange:
    """
    Typed wrapper around a ccxt exchange instance. Constructed once by
    trading_bot.py and injected into both agents.

    Portability note: most of this class is generic ccxt usage and
    exchange-agnostic. A few methods rely on Binance-specific ccxt
    behavior rather than a universal ccxt guarantee — notably
    create_market_buy_order()/create_market_sell_order()'s
    client_order_id forwarding and fetch_order_by_client_id()'s
    origClientOrderId-based lookup (see their own docstrings). This is
    intentional and contained: exchange.py is the sole ccxt translation
    boundary by design, so if this project ever targets a different
    exchange, these are the only methods that would need adapting —
    no other module depends on the underlying mechanism, only on this
    class's method contracts.
    """

    def __init__(
        self,
        exchange_id: str,
        api_key: str,
        api_secret: str,
        options: Optional[Dict[str, Any]] = None,
    ):
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ExchangeError(f"Unknown ccxt exchange id: {exchange_id}")

        default_options = {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
        }
        if options:
            default_options.update(options)

        try:
            self._exchange = exchange_class({
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": default_options,
            })
        except Exception as exc:
            raise ExchangeError(f"Failed to initialize exchange {exchange_id}: {exc}") from exc

        logger.info("Connected to %s exchange", exchange_id)

    # ── Market data ──────────────────────────────────────────────────────

    def load_markets(self, reload: bool = True) -> Dict[str, ExchangeMarket]:
        try:
            raw_markets = self._exchange.load_markets(reload=reload)
        except Exception as exc:
            raise ExchangeError(f"Failed to load markets: {exc}") from exc

        result: Dict[str, ExchangeMarket] = {}
        for symbol, market in raw_markets.items():
            if not isinstance(market, dict):
                logger.warning("Skipping malformed market entry for %s", symbol)
                continue

            active = market.get("active", True)
            active = True if active is None else bool(active)

            info = market.get("info", {}) or {}

            result[symbol] = ExchangeMarket(
                symbol=symbol,
                base=str(market.get("base", "")),
                quote=str(market.get("quote", "")),
                spot=bool(market.get("spot", False)),
                active=active,
                min_notional=self._extract_min_notional(info),
                max_notional=self._extract_max_notional(info),
                amount_precision=self._extract_amount_precision(market),
            )

        return result

    def fetch_market_metrics(self, symbols: List[str]) -> Dict[str, MarketMetrics]:
        """
        ASSUMPTION (not repository-confirmed): derives quote_volume_24h and
        spread_pct from ccxt's fetch_tickers(). trading_bot.py never uses
        fetch_tickers() — validate this against real Binance ticker payloads
        before relying on it in production filtering.
        """
        try:
            raw_tickers = self._exchange.fetch_tickers(symbols)
        except Exception as exc:
            raise ExchangeError(f"Failed to fetch tickers: {exc}") from exc

        result: Dict[str, MarketMetrics] = {}
        for symbol, ticker in raw_tickers.items():
            if not isinstance(ticker, dict):
                continue

            quote_volume = ticker.get("quoteVolume")
            bid = ticker.get("bid")
            ask = ticker.get("ask")

            spread_pct = None
            if bid is not None and ask is not None and bid > 0:
                spread_pct = (float(ask) - float(bid)) / float(bid)

            result[symbol] = MarketMetrics(
                quote_volume_24h=float(quote_volume) if quote_volume is not None else None,
                spread_pct=spread_pct,
            )

        return result

    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            return ticker.get("last")
        except Exception as exc:
            logger.error("Price check failed for %s: %s", symbol, exc)
            return None

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1m", limit: int = 20
    ) -> List[OHLCVBar]:
        try:
            raw = self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as exc:
            raise ExchangeError(f"Failed to fetch OHLCV for {symbol}: {exc}") from exc

        bars: List[OHLCVBar] = []
        for row in raw:
            if len(row) < 6:
                logger.warning("Skipping malformed OHLCV row for %s: %s", symbol, row)
                continue
            bars.append(
                OHLCVBar(
                    timestamp_ms=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        return bars

    # ── Order execution ──────────────────────────────────────────────────

    def create_market_buy_order(
        self, symbol: str, amount: float, client_order_id: Optional[str] = None
    ) -> OrderResult:
        """
        client_order_id, if supplied, is forwarded as ccxt's unified
        `newClientOrderId` param. NOTE — portability: this project
        currently targets Binance, where ccxt's `newClientOrderId`
        param is well-supported for spot market orders. Not every ccxt
        exchange accepts or honors this param identically (some ignore
        it, some use a different param name, some cap it at a
        different length). If this project is ever pointed at a
        different exchange, this is the correct (and only) place to
        adapt that — exchange.py is the sole ccxt translation boundary
        by design; nothing above it (trading_agent/execution.py
        included) should need to change.
        """
        params: Dict[str, Any] = {"type": "spot", "recvWindow": 10000}
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        try:
            order = self._exchange.create_market_buy_order(symbol, amount, params=params)
        except Exception as exc:
            raise ExchangeError(f"Buy order failed for {symbol}: {exc}") from exc
        return self._translate_order(order, symbol, "buy")

    def create_market_sell_order(
        self, symbol: str, amount: float, client_order_id: Optional[str] = None
    ) -> OrderResult:
        """
        See create_market_buy_order()'s docstring — same
        client_order_id forwarding and the same Binance-specific
        portability note applies here.
        """
        params: Dict[str, Any] = {"type": "spot", "recvWindow": 10000}
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        try:
            order = self._exchange.create_market_sell_order(symbol, amount, params=params)
        except Exception as exc:
            raise ExchangeError(f"Sell order failed for {symbol}: {exc}") from exc
        return self._translate_order(order, symbol, "sell")

    def fetch_order_by_client_id(self, symbol: str, client_order_id: str) -> Optional[OrderResult]:
        """
        Looks up a single order by the caller-supplied client order id
        rather than the exchange-assigned order id — used to reconcile
        an execution attempt whose original response was ambiguous
        (e.g. a network timeout after the order reached the exchange).

        NOTE — Binance-specific, not a generic ccxt guarantee: this
        currently relies on Binance's `origClientOrderId` param on
        ccxt's unified `fetchOrder`. Not every exchange ccxt exposes
        an equivalent lookup-by-client-id capability, and
        exchanges that do may use a different param name or a
        different underlying mechanism entirely. If this project is
        ever pointed at a different exchange, this method is the
        correct (and only) place to implement that exchange's
        equivalent — exchange.py remains the sole ccxt translation
        boundary; callers (trading_agent/execution.py) only depend on
        this method's return contract (Optional[OrderResult]), never
        on how the lookup is performed underneath.

        Returns None if no matching order is found or the lookup itself
        fails — callers treat that as "cannot confirm", not as "did not
        happen", and fall back to their own failure handling.
        """
        try:
            order = self._exchange.fetch_order(
                None, symbol, params={"origClientOrderId": client_order_id}
            )
        except Exception as exc:
            logger.warning(
                "Order lookup by client_order_id failed for %s (client_order_id=%s): %s",
                symbol, client_order_id, exc,
            )
            return None

        if not isinstance(order, dict):
            return None
        return self._translate_order(order, symbol, str(order.get("side", "")).lower())

    def fetch_my_trades(self, symbol: str, since_ms: int) -> List[TradeFill]:
        try:
            raw_trades = self._exchange.fetch_my_trades(symbol, since=since_ms)
        except Exception as exc:
            raise ExchangeError(f"Failed to fetch trades for {symbol}: {exc}") from exc

        fills: List[TradeFill] = []
        for trade in raw_trades:
            if not isinstance(trade, dict):
                continue
            amount = trade.get("amount", trade.get("filled", 0)) or 0
            price = trade.get("price", 0) or 0
            fee_cost = (trade.get("fee") or {}).get("cost", 0) or 0
            timestamp = trade.get("timestamp")
            if timestamp is None:
                continue
            fills.append(
                TradeFill(
                    symbol=symbol,
                    side=str(trade.get("side", "")).lower(),
                    amount=float(amount),
                    price=float(price),
                    fee_cost=float(fee_cost),
                    timestamp_ms=int(timestamp),
                )
            )
        return fills

    def fetch_open_orders(self, symbol: str) -> List[OrderResult]:
        try:
            raw_orders = self._exchange.fetch_open_orders(symbol)
        except Exception as exc:
            raise ExchangeError(f"Failed to fetch open orders for {symbol}: {exc}") from exc

        return [self._translate_order(o, symbol, str(o.get("side", "")).lower())
                for o in raw_orders if isinstance(o, dict)]

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception as exc:
            logger.warning("Failed to cancel order %s for %s: %s", order_id, symbol, exc)
            return False

    def milliseconds(self) -> int:
        return self._exchange.milliseconds()

    def verify_credentials(self) -> None:
        """
        Confirms the configured API credentials actually authenticate,
        via a minimal read-only authenticated call (fetch_balance).
        Used exclusively by the deployment config layer's startup
        connectivity check (config/config.py::validate_exchange_
        connectivity()) — never called mid-trading-cycle, and never
        used for its return value (the balance itself is not exposed;
        nothing outside this method needs it — only whether the call
        succeeded).

        fetch_balance() is used rather than load_markets() specifically
        because load_markets() is a public endpoint that succeeds with
        no credentials at all — it would not catch invalid/revoked API
        keys. fetch_balance() requires a valid signed request, so it
        fails immediately (before this method returns) on bad
        credentials, matching this method's sole purpose.

        Raises:
            ExchangeError: if authentication fails for any reason
                (invalid key/secret, revoked key, IP restriction, etc.).
        """
        try:
            self._exchange.fetch_balance(params={"type": "spot"})
        except Exception as exc:
            raise ExchangeError(f"Credential verification failed: {exc}") from exc

    # ── Internal translation helpers (ccxt-specific, private) ──────────────

    def _translate_order(self, order: Dict[str, Any], symbol: str, side: str) -> OrderResult:
        filled = order.get("filled", order.get("amount", 0)) or 0
        average = order.get("average", order.get("price", 0)) or 0
        fee_cost = (order.get("fee") or {}).get("cost", 0) or 0
        client_order_id = order.get("clientOrderId")
        return OrderResult(
            order_id=str(order.get("id", "")),
            symbol=symbol,
            side=side,
            status=order.get("status"),
            filled_amount=float(filled),
            average_price=float(average),
            fee_cost=float(fee_cost),
            client_order_id=str(client_order_id) if client_order_id else None,
        )

    def _extract_min_notional(self, market_info: Dict[str, Any]) -> float:
        """
        Repository logic ported from trading_bot.py's extract_min_notional().
        """
        try:
            for f in market_info.get("filters", []):
                if f.get("filterType") in ["MIN_NOTIONAL", "NOTIONAL"]:
                    val = f.get("minNotional") or f.get("min_notional") or f.get("notional", "0")
                    try:
                        return float(val)
                    except Exception:
                        return 0.0
            if "minNotional" in market_info:
                try:
                    return float(market_info["minNotional"])
                except Exception:
                    pass
            return 5.0
        except Exception as exc:
            logger.warning("Failed to extract minNotional: %s", exc)
            return 5.0

    def _extract_max_notional(self, market_info: Dict[str, Any]) -> float:
        try:
            return float(market_info.get("maxNotional", float("inf")))
        except Exception:
            return float("inf")

    def _extract_amount_precision(self, market: Dict[str, Any]) -> int:
        try:
            return int(market.get("precision", {}).get("amount", 6))
        except Exception:
            return 6