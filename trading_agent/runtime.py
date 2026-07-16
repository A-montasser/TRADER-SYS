"""
trading_agent/runtime.py

Runtime orchestrator for the Trading Agent.

Responsibility (frozen):
    Execute exactly one complete trading cycle using an existing
    Prediction Artifact: load, analyze, rank, assess risk (static and,
    if a position is open, live/dynamic), decide, allocate, execute,
    manage the position, journal, and (only when a trade closed this
    cycle) run meta-learning. Pure orchestration — no trading policy,
    no prediction logic.

    Performs one read-only market-price query per cycle (only when a
    position is open, to compute LiveRiskAssessment) via
    exchange.get_current_price() directly — this mirrors the
    established precedent of prediction_agent/runtime.py calling
    exchange.fetch_market_metrics() directly for read-only market data.
    This is distinct from execution.py's charter, which is order
    submission and sizing specifically, not general market-data reads.
    risk_manager.py and decision_engine.py never touch the exchange —
    they receive this price only as a plain computed metric
    (LiveRiskAssessment), never the raw value itself.

    Loads the artifact with expiry enforcement disabled whenever a
    Position is open — trading_bot.py may pass a Position Forecast
    (the exact artifact that justified that position's entry) whose
    validity window has since passed, which must remain loadable as
    that position's authoritative timeline reference until it closes.
    Integrity checks (schema version, artifact_id consistency) still
    apply regardless — only the freshness check is conditional.

Explicitly NOT this module's responsibility:
    - Any trading policy itself -> decision_engine.py, risk_manager.py,
      capital_manager.py, oppurtunity_ranker.py
    - Prediction logic            -> prediction_agent/
    - Scheduling / retry / monitoring / Forecast Cursor ownership -> trading_bot.py
    - Looping / sleeping            -> trading_bot.py
    - Order submission / sizing       -> trading_agent/execution.py

This module executes ONE cycle and returns. It does not loop, does not
sleep, does not retry, and does not own the Forecast Cursor or
open-position state — both are supplied by the caller and returned,
updated, in TradingCycleResult.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from exchange import Exchange
from models.trading import DecisionAction, Position, TradeDecision, TradeRecord

from trading_agent.artifact_loader import load_artifact
from trading_agent.scenario_analyzer import analyze_scenarios
from trading_agent.oppurtunity_ranker import rank_opportunities
from trading_agent.risk_manager import assess_risk, assess_live_risk
from trading_agent.decision_engine import decide
from trading_agent.capital_manager import allocate_capital
from trading_agent.execution import submit_buy_order, submit_sell_order, ExecutionError
from trading_agent.position_manager import create_position, close_position
from trading_agent.trade_journal import build_trade_record, persist_trade_record
from trading_agent.meta_learning import load_trade_history, analyze_symbol_performance

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradingCycleResult:
    """
    Module-local — consumed only by trading_bot.py (the top-level
    orchestrator, not another Trading Agent module), so it stays local
    rather than being promoted to models/trading.py, per the
    established "only boundary-crossing types go in models/" rule.

    Carries everything trading_bot.py needs to update its own runtime
    state (open_position, available_budget) between cycles.
    """
    decision: TradeDecision
    open_position: Optional[Position]
    available_budget: float
    trade_record: Optional[TradeRecord]


def run_trading_cycle(
    exchange: Exchange,
    parquet_path: Path,
    meta_path: Path,
    cursor: int,
    open_position: Optional[Position],
    available_budget: float,
    initial_balance: float,
    trade_history_path: Path,
) -> TradingCycleResult:
    """
    Executes one complete trading cycle.

    Args:
        exchange: instance of exchange.Exchange.
        parquet_path / meta_path: the current Prediction Artifact's files.
        cursor: current Forecast Cursor position — owned by trading_bot.py,
            supplied here, never stored.
        open_position: the currently open Position, if any — owned by
            trading_bot.py.
        available_budget: current Trading Agent budget.
        initial_balance: starting budget, used only to normalize
            meta-learning's performance_score.
        trade_history_path: trade journal CSV file.

    Returns:
        TradingCycleResult
    """
    artifact = load_artifact(parquet_path, meta_path, enforce_expiry=open_position is None)
    opportunities = analyze_scenarios(artifact)
    qualified = rank_opportunities(opportunities)
    risk_assessments = {o.symbol: assess_risk(o) for o in qualified}

    live_risk = None
    if open_position is not None and open_position.artifact_id == artifact.artifact_id:
        held_opportunity = next((o for o in qualified if o.symbol == open_position.symbol), None)
        if held_opportunity is not None:
            current_price = exchange.get_current_price(open_position.symbol)
            if current_price and current_price > 0:
                live_risk = assess_live_risk(held_opportunity, cursor, current_price)
            else:
                logger.warning(
                    "Runtime: no current price available for %s — proceeding without live risk data",
                    open_position.symbol,
                )

    decision = decide(
        qualified,
        cursor=cursor,
        open_position=open_position,
        risk_assessments=risk_assessments,
        current_artifact_id=artifact.artifact_id,
        horizon_bars=artifact.pred_len,
        live_risk=live_risk,
    )
    logger.info("Runtime: decision=%s reason=%s", decision.action, decision.reason)

    new_open_position = open_position
    new_budget = available_budget
    trade_record: Optional[TradeRecord] = None

    if decision.action == DecisionAction.ENTRY and decision.opportunity is not None:
        symbol = decision.opportunity.symbol
        allocated = allocate_capital(decision, new_budget, open_position=new_open_position)

        if allocated > 0:
            try:
                buy_result = submit_buy_order(exchange, symbol, allocated, artifact.artifact_id, cursor)
            except ExecutionError as exc:
                logger.error("Runtime: buy order failed for %s: %s", symbol, exc)
            else:
                assessment = risk_assessments[symbol]
                stop_loss = buy_result.average_price * (1 - assessment.stop_loss_pct / 100)
                take_profit = buy_result.average_price * (1 + assessment.take_profit_pct / 100)
                new_open_position = create_position(
                    buy_result,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    allocated_balance=allocated,
                    remaining_balance=new_budget - allocated,
                    artifact_id=artifact.artifact_id,
                    entry_bar=cursor,
                )
                new_budget = new_budget - allocated

    elif decision.action == DecisionAction.EXIT and open_position is not None:
        try:
            sell_result = submit_sell_order(
                exchange, open_position.symbol, open_position.amount, artifact.artifact_id, cursor
            )
        except ExecutionError as exc:
            logger.error("Runtime: sell order failed for %s: %s", open_position.symbol, exc)
        else:
            # Sell Commit Boundary: the exchange has now confirmed the
            # sell filled — the position IS closed from the exchange's
            # perspective, unconditionally, regardless of what happens
            # next. Internal state must never diverge from that fact:
            # re-attempting a sell next cycle for an asset no longer
            # held would be wrong. Everything below this point is
            # bookkeeping, not a reason to keep believing the position
            # is still open if it fails.
            new_open_position = None
            pnl = (sell_result.average_price - open_position.entry_price) * open_position.amount
            new_budget = open_position.remaining_balance + open_position.allocated_balance + pnl

            try:
                close_position(open_position, sell_result)
                trade_record = build_trade_record(open_position, sell_result, exit_reason=decision.reason)
                persist_trade_record(trade_record, trade_history_path)
                new_budget = trade_record.new_balance  # authoritative once journaling succeeds
            except Exception as exc:
                trade_record = None
                logger.critical(
                    "Runtime: sell for %s filled on the exchange but post-exit bookkeeping failed (%s) — "
                    "position closed internally with reconciled budget %.8f; this trade may be missing "
                    "from trade_history and requires manual reconciliation.",
                    open_position.symbol, exc, new_budget,
                )
            else:
                # Meta-learning only has new information to analyze once a
                # trade has actually been journaled — running it on every
                # WAIT/HOLD cycle would recompute identical results for
                # no reason.
                history = load_trade_history(trade_history_path)
                performance = analyze_symbol_performance(history, initial_balance=initial_balance)
                for p in performance:
                    logger.info(
                        "Runtime meta-learning: %s trades=%d win_rate=%.2f%% score=%.4f",
                        p.symbol, p.trade_count, p.win_rate * 100, p.performance_score,
                    )

    return TradingCycleResult(
        decision=decision,
        open_position=new_open_position,
        available_budget=new_budget,
        trade_record=trade_record,
    )