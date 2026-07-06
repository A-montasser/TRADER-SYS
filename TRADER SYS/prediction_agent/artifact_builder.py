"""
prediction_agent/artifact_builder.py

Prediction Artifact Generation module for the Prediction Agent.

Responsibility (frozen):
    Build the immutable PredictionArtifact and persist it to disk as
    the official inter-agent contract (parquet + meta.json).

Explicitly NOT this module's responsibility:
    - Kronos inference        -> prediction_agent/kronos_wrapper.py
    - Ranking                 -> prediction_agent/ranking.py
    - Forecast analytics       -> prediction_agent/analytics.py (internal only)
    - Trading decisions        -> trading/
    - Artifact loading/reading -> trading/artifact_loader.py
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import pandas as pd

from models.market import RankedCandidate
from models.artifact import ForecastResult, PredictionRecord, PredictionArtifact

logger = logging.getLogger(__name__)

ARTIFACT_SCHEMA_VERSION = 1


class ArtifactBuilderError(Exception):
    """Raised when a Prediction Artifact cannot be assembled or saved."""


@dataclass(frozen=True)
class ArtifactBuilderConfig:
    """
    Mirrors prediction_agent.yaml artifact parameters.
    Owned by artifact_builder.py — single consumer, constructed by
    runtime.py from parsed YAML.
    """
    validity_window: timedelta
    engine_reference: str


def build_artifact(
    forecast_results: tuple[ForecastResult, ...],
    ranked_candidates: List[RankedCandidate],
    config: ArtifactBuilderConfig,
) -> PredictionArtifact:
    """
    Assembles one PredictionArtifact for the current Prediction Cycle.

    Raises:
        ArtifactBuilderError: if no records could be assembled at all.
    """
    ranked_by_symbol: Dict[str, RankedCandidate] = {
        rc.candidate.symbol: rc for rc in ranked_candidates
    }

    records: List[PredictionRecord] = []
    for result in forecast_results:
        ranked = ranked_by_symbol.get(result.symbol)
        if ranked is None:
            logger.warning(
                "Skipping %s: no matching RankedCandidate found", result.symbol
            )
            continue

        records.append(
            PredictionRecord(
                symbol=result.symbol,
                forecast=result.forecast,
                ranking_position=ranked.rank,
                ranking_score=ranked.momentum_factor,
            )
        )

    if not records:
        raise ArtifactBuilderError(
            "No PredictionRecords could be assembled — empty forecast/ranking input "
            "or complete symbol mismatch"
        )

    generated_at = datetime.utcnow()
    artifact = PredictionArtifact(
        artifact_id=uuid4(),
        generated_at=generated_at,
        valid_from=generated_at,
        valid_until=generated_at + config.validity_window,
        engine_reference=config.engine_reference,
        records=tuple(records),
    )

    logger.info(
        "Artifact %s built: %d/%d records assembled, valid until %s",
        artifact.artifact_id, len(records), len(forecast_results), artifact.valid_until,
    )
    return artifact


def _flatten_records(artifact: PredictionArtifact) -> pd.DataFrame:
    """
    Flattens artifact.records into one row per (symbol, bar).
    ranking_position/ranking_score are denormalized (repeated) across
    all bar rows for a given symbol.
    """
    rows = []
    for record in artifact.records:
        for bar in record.forecast.bars:
            rows.append({
                "symbol": record.symbol,
                "ranking_position": record.ranking_position,
                "ranking_score": record.ranking_score,
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "amount": bar.amount,
            })
    return pd.DataFrame(rows)


def _serialize_parquet(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _stage_temp_file(path: Path, data: bytes) -> Path:
    """
    Writes data to a temp file in the target directory without
    replacing the production file. Returns the temp file path.
    """
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return Path(tmp_path)


def save_artifact(
    artifact: PredictionArtifact,
    parquet_path: Path,
    meta_path: Path,
) -> None:
    """
    Persists the artifact to disk as the official inter-agent contract:
    parquet (per-bar data) + meta.json (cycle metadata, including
    artifact_schema_version).

    Both files are fully staged (serialized + written to temp) before
    either production file is replaced, so a serialization failure
    never touches the existing artifact. Commit order is parquet
    first, meta.json last: if the process crashes between the two
    commits, the surviving old meta.json's valid_until still governs,
    and artifact_loader.py's expiry check naturally rejects the
    mismatched state rather than accepting stale data under a fresh
    validity window.

    Raises:
        ArtifactBuilderError: if serialization or writing fails.
    """
    parquet_path = Path(parquet_path)
    meta_path = Path(meta_path)

    try:
        df = _flatten_records(artifact)
        parquet_bytes = _serialize_parquet(df)
    except Exception as exc:
        raise ArtifactBuilderError(f"Failed to serialize artifact records: {exc}") from exc

    meta = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": str(artifact.artifact_id),
        "generated_at": artifact.generated_at.isoformat(),
        "valid_from": artifact.valid_from.isoformat(),
        "valid_until": artifact.valid_until.isoformat(),
        "engine_reference": artifact.engine_reference,
    }
    try:
        meta_bytes = json.dumps(meta, indent=2).encode("utf-8")
    except Exception as exc:
        raise ArtifactBuilderError(f"Failed to serialize artifact metadata: {exc}") from exc

    parquet_tmp = None
    meta_tmp = None
    try:
        parquet_tmp = _stage_temp_file(parquet_path, parquet_bytes)
        meta_tmp = _stage_temp_file(meta_path, meta_bytes)
    except Exception as exc:
        for tmp in (parquet_tmp, meta_tmp):
            if tmp is not None and tmp.exists():
                tmp.unlink()
        raise ArtifactBuilderError(f"Failed to stage artifact files: {exc}") from exc

    try:
        os.replace(parquet_tmp, parquet_path)
        os.replace(meta_tmp, meta_path)
    except Exception as exc:
        raise ArtifactBuilderError(f"Failed to commit artifact files: {exc}") from exc

    logger.info(
        "Artifact %s saved (schema v%d): %s, %s",
        artifact.artifact_id, ARTIFACT_SCHEMA_VERSION, parquet_path, meta_path,
    )