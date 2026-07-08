"""
models/trading.py

Shared Trading Agent data contracts that cross module boundaries
(scenario_analyzer.py -> opportunity_ranker.py -> decision_engine.py;
position_manager.py -> risk_manager.py -> execution.py -> trade_journal.py
-> meta_learning.py). No Kronos dependency.

These are data-only contracts. No business logic belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from models.artifact import PredictionRecord


@dataclass(frozen=True)
class Opportunity:
    """
    One symbol's trading scenario for the current Prediction Cycle,
    produced by scenario_analyzer.py directly from a PredictionRecord
    (models/artifact.py). Consumed by opportunity_ranker.py to select
    and order candidates for decision_engine.py.

    Opportunity is an enriched view of a PredictionRecord, not a
    compressed replacement of it: `record` holds a direct reference to
    the original PredictionRecord (and, through it, its ForecastSeries)
    so the Trading Agent retains access to the full predicted path
    throughout the artifact's lifetime — the forecast bars are never
    duplicated here.

    symbol/ranking_position/ranking_score are kept as top-level fields
    (in addition to being reachable via `record`) because
    opportunity_ranker.py already depends on them directly; this is a
    deliberate, minimal duplication rather than a data-modeling default.

    The remaining fields are deterministic, descriptive statistics of
    the record's ForecastSeries only — no trading policy (entry/exit
    thresholds, position sizing, live price) is embedded here; that
    belongs to decision_engine.py / risk_manager.py.
    """
    record: PredictionRecord
    symbol: str
    ranking_position: int
    ranking_score: float
    forecast_return_pct: float
    max_predicted_high: float
    min_predicted_low: float
    expected_range_pct: float
    drawdown_estimate_pct: float
    upside_estimate_pct: float


@dataclass(frozen=True)
class TradeDecision:
    """
    Output of decision_engine.py: whether the Trading Agent should open
    a trade this cycle, and if so, which qualified Opportunity was
    selected. Crosses decision_engine.py -> position_manager.py /
    capital_manager.py / execution.py, hence a shared contract.

    No sizing, no stop-loss/take-profit, no exit timing — those are
    computed by later modules once a trade is decided.
    """
    should_trade: bool
    opportunity: Optional[Opportunity]
    reason: str


@dataclass(frozen=True)
class RiskAssessment:
    """
    Cost-adjusted, forecast-path-derived metrics for one Opportunity,
    produced by risk_manager.py. Consumed by decision_engine.py, which
    owns the final trade/no-trade decision — this contract contains no
    decision, recommendation, or approval field.

    Contains only new information not already on Opportunity (which
    already owns forecast_return_pct, drawdown_estimate_pct,
    ranking_position, ranking_score, record).
    """
    estimated_fee_pct: float
    estimated_slippage_pct: float
    net_expected_profit_pct: float
    reward_to_risk_ratio: float
    bars_to_profitability: Optional[int]


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