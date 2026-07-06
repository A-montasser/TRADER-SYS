"""
models/artifact.py

Shared Prediction Artifact domain contracts — the sole communication
boundary between the Prediction Agent and the Trading Agent.

Engine-independent by design: no class here may reference Kronos or any
other specific forecasting engine. If the prediction engine is replaced,
this schema and the Trading Agent must remain unchanged.

These are data-only contracts. No business logic, no analytics, no
trading decisions, no serialization code belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID

# Schema version of the PredictionArtifact contract defined in this module.
# Owned here (not by artifact_builder.py or artifact_loader.py) because the
# artifact is the sole inter-agent contract: neither the Prediction Agent
# (which serializes it) nor the Trading Agent (which loads it) owns the
# contract itself.
ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PredictedBar:
    """
    One predicted future market bar. Produced by kronos_wrapper.py (or
    any future prediction engine wrapper), consumed by analytics.py and
    artifact_builder.py.
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float


@dataclass(frozen=True)
class ForecastSeries:
    """
    The complete predicted future time series for one symbol.
    Contains only forecast data — no symbol, no ranking, no trading
    context. Always nested inside a PredictionRecord, which supplies
    the symbol identity.
    """
    bars: tuple[PredictedBar, ...]


@dataclass(frozen=True)
class PredictionRecord:
    """
    One predicted market symbol within a Prediction Cycle.

    ranking_position / ranking_score are intentionally generic — they
    must never expose ranking-strategy-specific values (e.g. a momentum
    factor), so the artifact schema survives changes to ranking.py's
    internal strategy.
    """
    symbol: str
    forecast: ForecastSeries
    ranking_position: int
    ranking_score: float


@dataclass(frozen=True)
class PredictionArtifact:
    """
    One complete Prediction Cycle — the frozen inter-agent contract.

    artifact_id uniquely identifies this Prediction Cycle so that
    trade_journal.py, meta_learning.py, replay, and debugging tools can
    reference a specific cycle without relying on timestamps.

    valid_from / valid_until are run-level metadata: every symbol in one
    predict_batch() call shares an identical prediction window (Kronos
    enforces equal sequence/horizon lengths across the batch), so
    validity belongs to the artifact, not to each ForecastSeries.

    engine_reference identifies which model/weights produced this
    artifact (for auditability/reproducibility), without naming the
    engine in the type system itself.
    """
    artifact_id: UUID
    generated_at: datetime
    valid_from: datetime
    valid_until: datetime
    engine_reference: Optional[str]
    records: tuple[PredictionRecord, ...]


@dataclass(frozen=True)
class ForecastResult:
    symbol: str
    forecast: ForecastSeries