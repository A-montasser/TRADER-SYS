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

Known limitation (Stage 2 Integration Backlog — not resolved here):
    No fallback reconciliation via exchange.fetch_my_trades() is
    possible for ambiguous responses. TradeFill (models/execution.py)
    has no order_id field, so a TradeFill can never be turned into a
    complete OrderResult without inventing a placeholder order_id,
    which is not acceptable. This is a Stage 2 (models/execution.py)
    limitation, out of scope for Stage 3 to fix.
"""

from __future__ import annotations

import logging

from exchange import Exchange, ExchangeError
from models.execution import OrderResult

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Raised when an order cannot be confirmed as filled."""


def _is_filled(order: OrderResult) -> bool:
    return order.status == "closed" and order.filled_amount > 0


def submit_buy_order(exchange: Exchange, symbol: str, capital: float) -> OrderResult:
    """
    Converts `capital` (quote currency, e.g. USDT) into a base-asset
    quantity via the current exchange price, then submits a single
    market buy order attempt. Price lookup lives here, not in
    trading_agent/runtime.py, because converting a Trading Agent
    request (allocated capital) into an exchange operation (an order
    at a quantity) is exactly this module's charter as the sole
    exchange interaction layer.

    Args:
        exchange: instance of exchange.Exchange.
        symbol: market symbol to buy.
        capital: amount of quote currency to spend.

    Returns:
        OrderResult

    Raises:
        ExecutionError: if no current price is available, the order
            fails, or it does not come back filled. No retry is
            attempted — the caller decides recovery.
    """
    current_price = exchange.get_current_price(symbol)
    if not current_price or current_price <= 0:
        raise ExecutionError(f"No current price available for {symbol}")

    quantity = capital / current_price

    try:
        order = exchange.create_market_buy_order(symbol, quantity)
    except ExchangeError as exc:
        raise ExecutionError(f"Buy order failed for {symbol}: {exc}") from exc

    if not _is_filled(order):
        raise ExecutionError(
            f"Buy order for {symbol} not filled: status={order.status}, "
            f"filled_amount={order.filled_amount}"
        )

    logger.info(
        "Buy order filled: %s capital=%.8f price=%.8f amount=%.8f avg_price=%.8f fee=%.8f",
        symbol, capital, current_price, order.filled_amount, order.average_price, order.fee_cost,
    )
    return order


def submit_sell_order(exchange: Exchange, symbol: str, amount: float) -> OrderResult:
    """
    Submits a single market sell order attempt and returns the result
    if confirmed filled.

    Args:
        exchange: instance of exchange.Exchange.
        symbol: market symbol to sell.
        amount: quantity to sell.

    Returns:
        OrderResult

    Raises:
        ExecutionError: if the order fails or does not come back filled.
            No retry is attempted — the caller decides recovery.
    """
    try:
        order = exchange.create_market_sell_order(symbol, amount)
    except ExchangeError as exc:
        raise ExecutionError(f"Sell order failed for {symbol}: {exc}") from exc

    if not _is_filled(order):
        raise ExecutionError(
            f"Sell order for {symbol} not filled: status={order.status}, "
            f"filled_amount={order.filled_amount}"
        )

    logger.info(
        "Sell order filled: %s amount=%.8f avg_price=%.8f fee=%.8f",
        symbol, order.filled_amount, order.average_price, order.fee_cost,
    )
    return order