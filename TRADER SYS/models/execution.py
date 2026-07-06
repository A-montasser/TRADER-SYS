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
    """
    order_id: str
    symbol: str
    side: str
    status: Optional[str]
    filled_amount: float
    average_price: float
    fee_cost: float


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