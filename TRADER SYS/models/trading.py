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
from enum import Enum
from typing import Optional
from uuid import UUID

from models.artifact import PredictionRecord


@dataclass(frozen=True)
class ForecastWindow:
    """
    A bar-index range within a forecast path. 0-indexed, inclusive on
    both ends, matching PredictedBar tuple indices within
    ForecastSeries.bars. Shared by Opportunity's entry_window and
    profit_window — avoids duplicating the same (start, end) shape
    as separate primitive field pairs.
    """
    start_bar: int
    end_bar: int


@dataclass(frozen=True)
class Opportunity:
    """
    One symbol's trading scenario for the current Prediction Cycle,
    produced by scenario_analyzer.py directly from a PredictionRecord
    (models/artifact.py). Consumed by oppurtunity_ranker.py to select
    and order candidates for decision_engine.py.

    Opportunity is immutable and represents the complete offline
    analysis of one forecast — it never changes. Only the relationship
    between a runtime Forecast Cursor (owned by trading_bot.py, never
    stored here) and this Opportunity changes over time. entry_window/
    profit_window are that analysis's timing results: bar-index ranges
    describing when, within the forecast path, entering or realizing
    profit is favorable — purely descriptive, computed once, never a
    READY/WAIT/HOLD/EXIT flag (those are runtime decisions owned
    exclusively by decision_engine.py).

    Opportunity is an enriched view of a PredictionRecord, not a
    compressed replacement of it: `record` holds a direct reference to
    the original PredictionRecord (and, through it, its ForecastSeries)
    so the Trading Agent retains access to the full predicted path
    throughout the artifact's lifetime — the forecast bars are never
    duplicated here.

    symbol/ranking_position/ranking_score are kept as top-level fields
    (in addition to being reachable via `record`) because
    oppurtunity_ranker.py already depends on them directly; this is a
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
    entry_window: ForecastWindow
    profit_window: ForecastWindow


class DecisionAction(Enum):
    """
    The four states decision_engine.py can produce. Owned exclusively
    by decision_engine.py — no other module may reference or set these
    as anything other than reading a TradeDecision.action it received.
    """
    ENTRY = "ENTRY"
    WAIT = "WAIT"
    HOLD = "HOLD"
    EXIT = "EXIT"


@dataclass(frozen=True)
class TradeDecision:
    """
    Output of decision_engine.py: the timeline-aware trading decision
    for this cycle. Crosses decision_engine.py -> capital_manager.py /
    position_manager.py / execution.py, hence a shared contract.

    should_trade is retained (True only when action is ENTRY) solely
    for capital_manager.py's existing, unmodified interface — it is a
    derived convenience, not a second source of truth; `action` is
    authoritative.

    No sizing, no stop-loss/take-profit, no exit timing computation —
    those are computed by later modules once a decision is made.
    """
    action: DecisionAction
    should_trade: bool
    opportunity: Optional[Opportunity]
    reason: str


@dataclass(frozen=True)
class RiskAssessment:
    """
    Cost-adjusted, forecast-path-derived metrics for one Opportunity,
    produced by risk_manager.py. Consumed by decision_engine.py, which
    owns the final trade/no-trade decision — this contract contains no
    decision, recommendation, or approval field. Independent of the
    Forecast Cursor — always computed relative to the whole path.

    Contains only new information not already on Opportunity (which
    already owns forecast_return_pct, drawdown_estimate_pct,
    ranking_position, ranking_score, record).

    stop_loss_pct/take_profit_pct are expressed as percentages (not
    absolute prices) since RiskAssessment is computed before entry,
    when the actual fill price is not yet known — the caller converts
    to absolute price levels once a fill occurs (position_manager.py).
    They directly reuse Opportunity's own drawdown_estimate_pct/
    upside_estimate_pct rather than an independently invented figure —
    "how much adverse/favorable deviation this specific forecast
    already considered its own worst/best case."
    """
    estimated_fee_pct: float
    estimated_slippage_pct: float
    net_expected_profit_pct: float
    reward_to_risk_ratio: float
    bars_to_profitability: Optional[int]
    stop_loss_pct: float
    take_profit_pct: float


@dataclass(frozen=True)
class LiveRiskAssessment:
    """
    Dynamic, live-price-derived risk metric for an open Position,
    produced by risk_manager.py, consumed by decision_engine.py.
    Computed fresh each cycle a position is open — unlike the static
    RiskAssessment (computed once per candidate Opportunity from the
    whole forecast path alone), this reflects how far the OBSERVED
    market has diverged from the PREDICTED trajectory at the current
    Forecast Cursor position.

    Contains only the one piece of information that requires live
    price. decision_engine.py compares this against the held
    Opportunity's own drawdown_estimate_pct/upside_estimate_pct
    (already available there) rather than duplicating thresholds here.
    """
    forecast_deviation_pct: float


@dataclass(frozen=True)
class Position:
    """
    An open trade held by the Trading Agent, produced/updated by
    position_manager.py from OrderResult (models/execution.py) at entry.
    Consumed by decision_engine.py (HOLD/EXIT timeline evaluation) and
    trade_journal.py (at close, to build a TradeRecord).

    artifact_id/entry_bar are the minimum context needed to relocate
    the Opportunity that justified this entry, without storing the
    full (immutable, already-available-elsewhere) Opportunity object
    on Position: decision_engine.py re-locates it by matching
    Position.symbol against the same ranked opportunities list it
    already receives every cycle (safe because Opportunity is static
    for the artifact's lifetime), guarded by artifact_id equality in
    case the active artifact has changed since entry. entry_bar (the
    Forecast Cursor value at entry) is kept for future analysis/
    journaling use even though decision_engine.py does not currently
    need it for the HOLD/EXIT check itself.

    Other fields limited to execution/lifecycle facts with direct
    precedent in trading_bot.py's open_positions[symbol].

    stop_loss/take_profit are intentionally informational/audit-trail
    fields, not the active exit mechanism — this project's hybrid
    decision philosophy uses forecast timeline (profit_window),
    hold-time, and live forecast-deviation checks
    (risk_manager.assess_live_risk / decision_engine.py) as the actual
    EXIT triggers, which supersede a static price-level check. These
    two fields record what a naive stop-loss/take-profit would have
    been at entry, computed from RiskAssessment.stop_loss_pct/
    take_profit_pct, for journaling/analysis purposes only.
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
    artifact_id: UUID
    entry_bar: int


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