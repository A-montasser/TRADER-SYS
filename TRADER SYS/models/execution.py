"""
models/execution.py

Shared order/trade execution contracts, produced by exchange.py and
consumed by trading/execution.py, trading/position_manager.py.

These are data-only contracts. No business logic belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OrderResult:
    """
    Translated order response. Field derivation matches trading_bot.py's
    enter_trade()/exit_trade() reads of the raw ccxt order dict.

    client_order_id: the caller-supplied identifier for this execution
    attempt (ccxt's unified 'clientOrderId' field), when the exchange
    returns/echoes one. Added so an ambiguous submission (e.g. a
    network timeout after the order reached the exchange) can be
    reconciled by looking the same attempt back up via
    Exchange.fetch_order_by_client_id() — production bug fix, not a
    new field for its own sake. Optional/defaulted because not every
    OrderResult originates from a client-order-id-tagged submission
    (e.g. results translated from fetch_open_orders()).
    """
    order_id: str
    symbol: str
    side: str
    status: Optional[str]
    filled_amount: float
    average_price: float
    fee_cost: float
    client_order_id: Optional[str] = None


@dataclass(frozen=True)
class TradeFill:
    """
    Translated trade-history entry. Field derivation matches
    trading_bot.py's reconcile_stale_position()/recover_from_failed_trade()
    reads of the raw ccxt trade dict.
    """
    symbol: str
    side: str
    amount: float
    price: float
    fee_cost: float
    timestamp_ms: int