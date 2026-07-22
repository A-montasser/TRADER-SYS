"""
trading_agent/execution.py

Execution module for the Trading Agent — the only module that
submits orders to the exchange on the Trading Agent's behalf.

Responsibility (frozen):
    Submit one BUY or SELL market order via exchange.py, confirm fill
    status, and return OrderResult (models/execution.py) — or raise
    ExecutionError honestly on failure. A pure translation boundary
    between Trading Agent order requests and exchange.py — no
    Binance-specific logic here, all of that already lives in
    exchange.py.

    This module's charter is order submission and sizing specifically
    — not "the only module that ever reads from the exchange." Read-
    only market-data queries for orchestration/analysis purposes (e.g.
    trading_agent/runtime.py reading current price for live risk
    assessment, mirroring prediction_agent/runtime.py reading market
    metrics) are a distinct concern and are not routed through here;
    routing them through this module would make it a valueless pass-
    through wrapper rather than meaningfully own anything.

    Buy orders are requested in quote-currency capital (e.g. USDT);
    this module looks up the current price and converts to a base-asset
    quantity itself, since that conversion is part of translating a
    capital allocation into an exchange operation — this module's own
    charter, not orchestration's. Sell orders are requested directly
    in base-asset quantity (the caller already knows exactly how much
    it holds — no conversion needed).

    Exactly one execution attempt per call. No retry loop, no
    time.sleep(), no internal waiting or recovery scheduling —
    retry/skip/recovery strategy belongs exclusively to trading_bot.py's
    runtime orchestration. This module only executes once and reports
    the outcome; it never decides what to do about a failure.

Explicitly NOT this module's responsibility:
    - Trade / hold / exit decision   -> trading_agent/decision_engine.py
    - Opportunity evaluation           -> trading_agent/scenario_analyzer.py, oppurtunity_ranker.py
    - Risk evaluation                    -> trading_agent/risk_manager.py
    - Capital allocation                  -> trading_agent/capital_manager.py
    - Position lifecycle                   -> trading_agent/position_manager.py
    - Trade journaling                      -> trading_agent/trade_journal.py
    - Retry / recovery strategy               -> trading_bot.py
    - Prediction Agent work                     -> never imported here

Resolved (was: Stage 2 Integration Backlog):
    Ambiguous-response reconciliation is implemented via a client
    order id generated fresh per attempt (see _build_client_order_id())
    and Exchange.fetch_order_by_client_id() — this sidesteps the
    original blocker entirely (TradeFill having no order_id) by
    looking the attempt up as an order, not a trade fill, so no
    models/execution.py change was needed for TradeFill. OrderResult
    did gain a client_order_id field (models/execution.py) to carry
    the id through to the caller.

    This module itself is exchange-agnostic — it only calls
    Exchange.fetch_order_by_client_id() and never touches ccxt/Binance
    specifics directly. The underlying lookup mechanism (Binance's
    origClientOrderId support via ccxt) is Binance-specific and lives
    entirely inside exchange.py — see that module's docstring for the
    portability note. If the exchange is ever changed, only exchange.py
    needs to change.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Optional
from uuid import UUID, uuid4

from exchange import Exchange, ExchangeError
from models.execution import OrderResult
from models.market import ExchangeMarket

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Raised when an order cannot be confirmed as filled."""


def _is_filled(order: OrderResult) -> bool:
    return order.status == "closed" and order.filled_amount > 0


def _quantize_amount(amount: float, precision: int) -> float:
    """
    Rounds down to the exchange's allowed amount precision. Floors
    rather than rounds nearest: for a buy, rounding up could request a
    quantity costing more than the allocated capital; for a sell,
    rounding up could request more than the position actually holds.
    Floor is the only direction that is safe in both directions.
    """
    factor = 10 ** precision
    return math.floor(amount * factor) / factor


def _validate_and_quantize(symbol: str, quantity: float, price: float, market: ExchangeMarket) -> float:
    """
    Enforces the exchange's own constraints (amount precision, min/max
    notional — already extracted by exchange.py onto ExchangeMarket)
    before an order is ever submitted, rather than letting the exchange
    reject it after the fact. Reuses exchange.py's existing metadata;
    does not duplicate its extraction logic.

    Raises:
        ExecutionError: if the quantized quantity is non-positive, or
            the resulting notional falls outside [min_notional, max_notional].
    """
    quantized = _quantize_amount(quantity, market.amount_precision)
    if quantized <= 0:
        raise ExecutionError(
            f"Quantity for {symbol} rounds to zero at exchange precision "
            f"{market.amount_precision} (requested={quantity:.8f})"
        )

    notional = quantized * price
    if notional < market.min_notional:
        raise ExecutionError(
            f"Order notional {notional:.8f} for {symbol} is below the exchange "
            f"minimum {market.min_notional:.8f}"
        )
    if notional > market.max_notional:
        raise ExecutionError(
            f"Order notional {notional:.8f} for {symbol} exceeds the exchange "
            f"maximum {market.max_notional:.8f}"
        )
    return quantized


def _get_market(exchange: Exchange, symbol: str) -> ExchangeMarket:
    """
    Sources market metadata via exchange.py's existing load_markets()
    — reload=False so this reuses ccxt's already-loaded market cache
    rather than issuing a fresh network call on every order attempt.
    """
    markets = exchange.load_markets(reload=False)
    market = markets.get(symbol)
    if market is None:
        raise ExecutionError(f"No exchange market metadata available for {symbol}")
    return market


def _build_client_order_id(artifact_id: UUID, symbol: str, side: str, cursor: int) -> str:
    """
    Client order id for one execution attempt.

    IMPORTANT — cursor is a forecast-bar index, not an execution
    attempt counter. It identifies a position within the Prediction
    Artifact's forecast path, not how many times an order has been
    attempted. More than one genuinely distinct execution attempt can
    occur for the same (artifact_id, symbol, side, cursor) — e.g. a
    second attempt after an ambiguous first response, a process
    restart, or any future retry policy — so cursor must never be
    relied on as the sole source of uniqueness: doing so could
    generate the same client order id for two different attempts,
    which the exchange would then reject (or, worse, wrongly conflate).

    (artifact_id, symbol, side, cursor) are embedded below only as a
    short, human-legible correlation tag — useful for tracing an order
    found in exchange history/logs back to the cycle and artifact that
    produced it. Uniqueness itself comes entirely from a fresh random
    component (`uuid4()`) generated at call time. This needs no
    persistent attempt-counter state — which, notably, would not even
    survive the restart scenario above, since process memory resets on
    restart while a fresh nonce does not need memory to stay unique.

    "Deterministic" here means only: the id is generated once per call
    and held for the lifetime of that call, so the exact same id is
    used both for the order submission and for the immediate
    reconciliation lookup if that submission's response turns out to
    be ambiguous (see _reconcile_ambiguous_order()) — it is a local
    value scoped to one submit_buy_order()/submit_sell_order()
    invocation, not a value meant to be independently recomputed later
    or across a restart.

    Hashed/truncated to stay within exchange client-order-id length
    limits (e.g. Binance's ~36 characters).
    """
    tag_raw = f"{artifact_id}:{symbol}:{side}:{cursor}"
    tag = hashlib.sha1(tag_raw.encode("utf-8")).hexdigest()[:12]
    nonce = uuid4().hex[:14]
    return f"ts{tag}{nonce}"


def _reconcile_ambiguous_order(
    exchange: Exchange, symbol: str, client_order_id: str, side: str
) -> Optional[OrderResult]:
    """
    Looks up the exact execution attempt identified by client_order_id
    to determine whether it actually reached and was accepted by the
    exchange, despite an ambiguous local response (an exception from
    the submission call, or a response that did not come back filled).

    Returns the OrderResult if found and it belongs to this side of
    this symbol; None if no matching order is found (or the lookup
    itself is inconclusive) — callers treat None as "still cannot
    confirm" and fall back to raising ExecutionError, never as proof
    the order didn't happen.
    """
    order = exchange.fetch_order_by_client_id(symbol, client_order_id)
    if order is None:
        return None
    if order.side != side:
        return None
    return order


def submit_buy_order(
    exchange: Exchange, symbol: str, capital: float, artifact_id: UUID, cursor: int
) -> OrderResult:
    """
    Converts `capital` (quote currency, e.g. USDT) into a base-asset
    quantity via the current exchange price, quantizes/validates it
    against exchange market constraints, then submits a single market
    buy order attempt tagged with a deterministic client order id.
    Price lookup lives here, not in trading_agent/runtime.py, because
    converting a Trading Agent request (allocated capital) into an
    exchange operation (an order at a quantity) is exactly this
    module's charter as the sole exchange interaction layer.

    Args:
        exchange: instance of exchange.Exchange.
        symbol: market symbol to buy.
        capital: amount of quote currency to spend.
        artifact_id: the Prediction Artifact this decision was made
            from — embedded in the client order id as a human-legible
            correlation tag (see _build_client_order_id()); not the
            source of its uniqueness.
        cursor: the Forecast Cursor (forecast-bar index) at the moment
            of this attempt — also embedded as correlation context
            only. It is NOT an execution attempt counter and must not
            be relied on to disambiguate attempts; see
            _build_client_order_id() for why.

    Returns:
        OrderResult

    Raises:
        ExecutionError: if no current price is available, the order
            fails and cannot be reconciled as having filled anyway, or
            it does not come back filled. No retry is attempted — the
            caller decides recovery.
    """
    current_price = exchange.get_current_price(symbol)
    if not current_price or current_price <= 0:
        raise ExecutionError(f"No current price available for {symbol}")

    market = _get_market(exchange, symbol)
    quantity = _validate_and_quantize(symbol, capital / current_price, current_price, market)
    client_order_id = _build_client_order_id(artifact_id, symbol, "buy", cursor)

    try:
        order = exchange.create_market_buy_order(symbol, quantity, client_order_id=client_order_id)
    except ExchangeError as exc:
        reconciled = _reconcile_ambiguous_order(exchange, symbol, client_order_id, "buy")
        if reconciled is not None and _is_filled(reconciled):
            logger.warning(
                "Buy order for %s (client_order_id=%s) reconciled as filled after an "
                "ambiguous submission response: %s", symbol, client_order_id, exc,
            )
            return reconciled
        raise ExecutionError(f"Buy order failed for {symbol}: {exc}") from exc

    if not _is_filled(order):
        reconciled = _reconcile_ambiguous_order(exchange, symbol, client_order_id, "buy")
        if reconciled is not None and _is_filled(reconciled):
            logger.warning(
                "Buy order for %s (client_order_id=%s) reconciled as filled after an "
                "initial not-filled response", symbol, client_order_id,
            )
            return reconciled
        raise ExecutionError(
            f"Buy order for {symbol} not filled: status={order.status}, "
            f"filled_amount={order.filled_amount}"
        )

    logger.info(
        "Buy order filled: %s capital=%.8f price=%.8f amount=%.8f avg_price=%.8f fee=%.8f "
        "client_order_id=%s",
        symbol, capital, current_price, order.filled_amount, order.average_price,
        order.fee_cost, client_order_id,
    )
    return order


def submit_sell_order(
    exchange: Exchange, symbol: str, amount: float, artifact_id: UUID, cursor: int
) -> OrderResult:
    """
    Quantizes/validates `amount` against exchange market constraints,
    then submits a single market sell order attempt tagged with a
    deterministic client order id, and returns the result if confirmed
    filled.

    Args:
        exchange: instance of exchange.Exchange.
        symbol: market symbol to sell.
        amount: quantity to sell.
        artifact_id / cursor: see submit_buy_order() — same
            deterministic client-order-id derivation.

    Returns:
        OrderResult

    Raises:
        ExecutionError: if the order fails and cannot be reconciled as
            having filled anyway, or does not come back filled. No
            retry is attempted — the caller decides recovery.
    """
    market = _get_market(exchange, symbol)
    current_price = exchange.get_current_price(symbol)
    if not current_price or current_price <= 0:
        raise ExecutionError(f"No current price available for {symbol}")

    quantity = _validate_and_quantize(symbol, amount, current_price, market)
    client_order_id = _build_client_order_id(artifact_id, symbol, "sell", cursor)

    try:
        order = exchange.create_market_sell_order(symbol, quantity, client_order_id=client_order_id)
    except ExchangeError as exc:
        reconciled = _reconcile_ambiguous_order(exchange, symbol, client_order_id, "sell")
        if reconciled is not None and _is_filled(reconciled):
            logger.warning(
                "Sell order for %s (client_order_id=%s) reconciled as filled after an "
                "ambiguous submission response: %s", symbol, client_order_id, exc,
            )
            return reconciled
        raise ExecutionError(f"Sell order failed for {symbol}: {exc}") from exc

    if not _is_filled(order):
        reconciled = _reconcile_ambiguous_order(exchange, symbol, client_order_id, "sell")
        if reconciled is not None and _is_filled(reconciled):
            logger.warning(
                "Sell order for %s (client_order_id=%s) reconciled as filled after an "
                "initial not-filled response", symbol, client_order_id,
            )
            return reconciled
        raise ExecutionError(
            f"Sell order for {symbol} not filled: status={order.status}, "
            f"filled_amount={order.filled_amount}"
        )

    logger.info(
        "Sell order filled: %s amount=%.8f avg_price=%.8f fee=%.8f client_order_id=%s",
        symbol, order.filled_amount, order.average_price, order.fee_cost, client_order_id,
    )
    return order