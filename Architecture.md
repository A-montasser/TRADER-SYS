# TRADER-SYS Architecture

Version: 2.0
Architecture Status: Frozen

---

# 1. Project Overview

TRADER-SYS is a fully autonomous AI cryptocurrency spot scalping system designed for production deployment.

The primary objective is maximizing autonomous trading performance while maintaining reliability, low latency, loose coupling and operational safety.

Kronos is used exclusively as the prediction engine.

---

# 2. Design Goals

- Spot Trading
- Halal Symbols
- Single Active Position
- Sequential Trading
- Continuous Compounding
- Low Latency
- Production Readiness
- Loose Coupling
- Fault Tolerance

---

# 3. High-Level Architecture

                    User
                      │
                      ▼
                trading_bot.py
                      │
        ┌─────────────┴─────────────┐
        │                           │
 Prediction Runtime          Trading Runtime
        │                           ▲
        └────────────┬──────────────┘
                     │
             Prediction Artifact

Prediction Artifact is the only communication contract.

---

# 4. Core Design Principles

- Prediction and trading are fully separated.
- Prediction Agent never makes trading decisions.
- Trading Agent never imports Kronos.
- Prediction Artifact is the only communication boundary.
- Runtime safety takes priority over implementation convenience.

---

# 5. System Components

## Prediction Agent

- Market scanning
- Market filtering
- Halal filtering
- Symbol ranking
- Market data acquisition
- Validation
- Kronos inference
- Forecast analytics
- Prediction Artifact generation
- Serialization

## Trading Agent

- Artifact loading
- Scenario analysis
- Opportunity ranking
- Decision making
- Risk management
- Position management
- Capital management
- Exchange execution
- Trade journal
- Meta learning

## Trading Bot

System orchestrator responsible for coordinating both runtimes and managing startup, shutdown and recovery.

## Exchange Layer

Provides the abstraction between the trading system and the exchange.

---

# 6. Prediction Artifact Contract

Prediction Artifact is the only shared interface between the Prediction Agent and the Trading Agent.

It contains prediction metadata and forecast data while remaining independent from Kronos implementation details.

---

# 7. Runtime Architecture

Prediction Runtime continuously generates Prediction Artifacts.

Trading Runtime continuously evaluates the latest valid artifact.

Trading Bot orchestrates the complete lifecycle.

---

# 8. Trading Lifecycle

Market Scan

↓

Prediction

↓

Prediction Artifact

↓

Scenario Analysis

↓

Opportunity Ranking

↓

Decision

↓

Execution

↓

Journal

↓

Repeat

---

# 9. Persistence & Recovery

- Artifact persistence
- Position persistence
- Restart recovery
- Graceful shutdown
- Runtime reconstruction
- Artifact cleanup
- Sell commit consistency

---

# 10. Repository Structure

project/

├── config/

├── artifacts/

├── models/

├── prediction_agent/

├── trading_agent/

├── exchange.py

└── trading_bot.py

---

# 11. Engineering Principles

- Repository is the implementation source of truth.
- Architecture defines system boundaries.
- Never introduce hidden coupling.
- Prefer production-quality implementations.
- Extend existing modules whenever possible.

---

# 12. Design Constraints

- Prediction Agent never executes trades.
- Trading Agent never imports Kronos.
- Prediction Artifact is the only communication contract.
- Single active position.
- Sequential execution.
- Spot trading only.
- Architecture changes require explicit approval.
