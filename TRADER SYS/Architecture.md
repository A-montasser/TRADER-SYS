# TRADER-SYS Architecture

Version: 1.0  
Status: Stage 3 (Trading Agent Development)  
Architecture Status: Frozen
Repository Status: Stage 2 Complete / Stage 3 In Progress

---

# Project Objective

TRADER-SYS is a fully autonomous AI cryptocurrency spot scalping system.

The objective is **maximizing autonomous trading performance**, not maximizing prediction accuracy.

Kronos is used exclusively as a prediction engine.

The system is designed around:

- Spot Trading
- Halal Symbols
- Single Active Position
- Sequential Trading
- Continuous Compounding
- Low Latency
- Production Readiness
- Loose Coupling

---

# Design Philosophy

The architecture separates prediction from trading.

Prediction generates information.

Trading makes decisions.

These responsibilities never overlap.

This separation guarantees:

- maintainability
- independent evolution
- production reliability
- easy testing
- minimal coupling

---

# High-Level Architecture

```
                    User
                      │
                      ▼
              trading_bot.py
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
 Prediction Agent          Trading Agent
        │                           ▲
        └────────────┬──────────────┘
                     │
             Prediction Artifact
```

Only the Prediction Artifact is shared.

No direct imports are allowed.

---

# Prediction Agent

Status:

✅ Complete

Frozen after Stage 2.

Responsibilities:

- Market Scanning
- Market Filtering
- Halal Filtering
- Symbol Ranking
- Market Data Download
- Validation
- Kronos Inference
- Internal Forecast Analytics
- Prediction Artifact Generation
- Prediction Artifact Serialization

Prediction Agent never:

- buys
- sells
- ranks trading opportunities
- calculates risk
- manages positions

Prediction Agent exports only Prediction Artifacts.

---

# Trading Agent

Status:

🚧 Under Development (Stage 3)

Responsibilities:

- Artifact Loading
- Scenario Analysis
- Opportunity Ranking
- Entry Decision
- Hold Management
- Exit Decision
- Risk Management
- Position Management
- Capital Management
- Exchange Execution
- Trade Journal
- Meta Learning

Trading Agent never:

- imports Kronos
- accesses prediction internals
- modifies prediction artifacts

---

# Inter-Agent Contract

Prediction Agent

↓

Prediction Artifact

↓

Trading Agent

The Prediction Artifact is the ONLY communication interface.

Nothing else crosses the boundary.

---

# Prediction Artifact

PredictionArtifact

Contains:

- artifact_id
- generated_at
- valid_from
- valid_until
- engine_reference
- records

Each record:

PredictionRecord

Contains:

- symbol
- forecast
- ranking_position
- ranking_score

Forecast:

ForecastSeries

Contains:

tuple[PredictedBar]

PredictedBar contains:

- timestamp
- open
- high
- low
- close
- volume
- amount

No analytics.

No confidence.

No recommendations.

No Kronos-specific objects.

---

# Serialization

Prediction Agent writes:

prediction_artifact.parquet

prediction_artifact.meta.json

Parquet:

Prediction rows.

Metadata:

Artifact metadata.

Trading Agent reconstructs the complete PredictionArtifact from both files.

Atomic persistence guarantees consistency.

---

# Repository Layout

```
project/

├── config/
│
├── artifacts/
│
├── models/
│
├── prediction_agent/
│
├── trading/
│
├── exchange.py
│
└── trading_bot.py
```

This document describes the approved target architecture.

The repository is developed incrementally.

Some modules listed below may be planned but not yet implemented.

This does not imply an architectural inconsistency.

Repository implementation may temporarily contain only a subset while development is in progress.

---

# Development Rules

Development follows the approved architecture.

Modules are implemented incrementally.

Implemented modules are considered stable unless a real integration issue is discovered.

Planned modules must follow this architecture exactly.

Architecture changes require explicit approval.

---

# Current Development Status

## Stage 1

✅ Complete

---

## Stage 2

✅ Complete

Includes:

- Kronos Integration
- Prediction Models
- Prediction Artifact
- Artifact Builder
- Analytics
- Serialization
- Schema Versioning
- Atomic Persistence

Prediction Agent is frozen.

---

## Stage 3

🚧 In Progress

Trading Agent implementation.

Development order:

1. artifact_loader.py
2. scenario_analyzer.py
3. opportunity_ranker.py
4. decision_engine.py
5. risk_manager.py
6. position_manager.py
7. capital_manager.py
8. execution.py
9. trade_journal.py
10. meta_learning.py

---

## Stage 4

Planned

Remaining implementation:

- prediction_agent/runtime.py
- trading_bot.py
- YAML configuration
- Final integration
- End-to-end testing

---

# # Implementation Status

The following table reflects the implementation status of the approved architecture.

| Module | Status |
|---------|--------|
| exchange.py | ✅ Implemented |
| models/market.py | ✅ Implemented |
| models/execution.py | ✅ Implemented |
| models/trading.py | ✅ Implemented |
| models/artifact.py | ✅ Implemented |
| prediction_agent/scanner.py | ✅ Implemented |
| prediction_agent/filters.py | ✅ Implemented |
| prediction_agent/ranking.py | ✅ Implemented |
| prediction_agent/downloader.py | ✅ Implemented |
| prediction_agent/validator.py | ✅ Implemented |
| prediction_agent/kronos_wrapper.py | ✅ Implemented |
| prediction_agent/analytics.py | ✅ Implemented |
| prediction_agent/artifact_builder.py | ✅ Implemented |
| prediction_agent/runtime.py | ⏳ Planned |
| trading/__init__.py | 🚧 Not Started |
| trading/artifact_loader.py | 🚧 Not Started |
| trading/scenario_analyzer.py | ⏳ Planned |
| trading/opportunity_ranker.py | ⏳ Planned |
| trading/decision_engine.py | ⏳ Planned |
| trading/risk_manager.py | ⏳ Planned |
| trading/position_manager.py | ⏳ Planned |
| trading/capital_manager.py | ⏳ Planned |
| trading/execution.py | ⏳ Planned |
| trading/trade_journal.py | ⏳ Planned |
| trading/meta_learning.py | ⏳ Planned |
| trading_bot.py | ⏳ Planned |
| config/*.yaml | ⏳ Planned |

---

# Engineering Principles

- Repository is the implementation source of truth.
- Architecture is the design source of truth.
- Never redesign architecture without explicit approval.
- Never introduce hidden coupling.
- Prefer simple production-quality implementations.
- Reuse Kronos whenever possible.
- Trading performance has priority over forecasting metrics.

---

# Development Workflow

Before implementing any module:

1. Review repository facts.
2. Validate architecture.
3. Identify implementation constraints.
4. Explain responsibilities.
5. Separate:
   - Repository Facts
   - Engineering Decisions
   - Assumptions
   - Recommendations
6. Obtain architectural agreement.
7. Implement.

---

# Repository First Policy

If repository implementation conflicts with previous conversations:

The repository wins.

If repository implementation conflicts with the frozen architecture:

Discuss before modifying.

Never silently redesign the architecture.

---

# Stage Completion Policy

Whenever a stage is completed:

- Update repository.
- Update ARCHITECTURE.md.
- Update project instructions if necessary.

Documentation must always reflect the current project state.

---

# Final Development Flow

Prediction Agent

Scanner
    ↓
Filters
    ↓
Ranking
    ↓
Downloader
    ↓
Validator
    ↓
Kronos Wrapper
    ↓
Analytics
    ↓
Artifact Builder
    ↓
Prediction Artifact
    ↓
Trading Agent
        ↓
Artifact Loader
        ↓
Scenario Analyzer
        ↓
Opportunity Ranker
        ↓
Decision Engine
        ↓
Risk Manager
        ↓
Position Manager
        ↓
Capital Manager
        ↓
Execution
        ↓
Trade Journal
        ↓
Meta Learning

---

# Source of Truth

Implementation details are defined by the repository.

System design is defined by this Architecture document.

If implementation conflicts with this document, the discrepancy must be reviewed before changing either one.

Neither should be modified implicitly.