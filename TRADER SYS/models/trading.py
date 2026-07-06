"""
models/trading.py

Shared Trading Agent data contracts that cross module boundaries
(position_manager.py -> risk_manager.py -> execution.py -> trade_journal.py
-> meta_learning.py). No Kronos dependency.

These are data-only contracts. No business logic belongs here.

NOTE: Opportunity is intentionally not yet defined here. It is produced by
scenario_analyzer.py/opportunity_ranker.py directly from PredictionRecord
(models/artifact.py, Stage 3 — not yet designed). Defining it now would
require inventing fields ahead of the Kronos-integration design, which is
explicitly disallowed. Add Opportunity once models/artifact.py is frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class Position:
    """
    An open trade held by the Trading Agent, produced/updated by
    position_manager.py from OrderResult (models/execution.py) at entry.
    Consumed by risk_manager.py (stop-loss/take-profit checks) and
    trade_journal.py (at close, to build a TradeRecord).

    Fields limited to execution/lifecycle facts with direct precedent in
    trading_bot.py's open_positions[symbol]. Prediction-context linkage
    (e.g. which Opportunity justified this entry) is deferred until
    models/artifact.py defines that type — not invented here.
    """
    symbol: str
    entry_price: float
    entry_time: datetime
    amount: float
    stop_loss: float
    take_profit: float
    order_id: str
    fee: float
    allocated_balance: float
    remaining_balance: float


@dataclass(frozen=True)
class TradeRecord:
    """
    A completed trade, produced by trade_journal.py from a closed
    Position plus its exit OrderResult/TradeFill. Persisted as the
    trade history source of truth. Consumed by meta_learning.py.

    Fields limited to execution/outcome facts with direct precedent in
    trading_bot.py's exit_trade() trade_result. The old meta-model
    feature columns (signal_strength, confidence, hybrid_score,
    predicted_return, dl_direction, dl_confidence, model_agreement,
    price_distance_pct) are excluded — they belong to the superseded
    LGBM/DL signal schema. meta_learning.py's Kronos-era feature needs
    will be defined once PredictionRecord (models/artifact.py) exists.
    """
    symbol: str
    entry_price: float
    exit_price: float
    amount: float
    pnl: float
    pnl_pct: float
    fee: float
    entry_time: datetime
    exit_time: datetime
    duration_min: float
    exit_reason: str
    new_balance: float