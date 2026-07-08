"""
trading/artifact_loader.py

Prediction Artifact Loading module for the Trading Agent.

Responsibility (frozen):
    Load prediction_artifact.parquet + prediction_artifact.meta.json,
    validate the pair for structural and cross-file integrity, and
    reconstruct the immutable PredictionArtifact object graph defined
    in models/artifact.py.

Explicitly NOT this module's responsibility:
    - Prediction Artifact generation / serialization -> prediction_agent/artifact_builder.py
    - Kronos inference                                -> prediction_agent/ (never imported here)
    - Scenario analysis / opportunity ranking          -> trading/scenario_analyzer.py, opportunity_ranker.py
    - Trading decisions                                -> trading/decision_engine.py and later modules
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pandas as pd

from models.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    ForecastSeries,
    PredictedBar,
    PredictionArtifact,
    PredictionRecord,
)

logger = logging.getLogger(__name__)

_REQUIRED_META_KEYS = (
    "artifact_schema_version",
    "artifact_id",
    "generated_at",
    "valid_from",
    "valid_until",
    "engine_reference",
    "pred_len",
    "timeframe",
)

_REQUIRED_PARQUET_COLUMNS = (
    "symbol",
    "ranking_position",
    "ranking_score",
    "artifact_id",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)


class ArtifactLoaderError(Exception):
    """Raised when a Prediction Artifact cannot be loaded or fails validation."""


def _load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        raise ArtifactLoaderError(f"Meta file not found: {meta_path}")

    try:
        raw = meta_path.read_text(encoding="utf-8")
        meta = json.loads(raw)
    except Exception as exc:
        raise ArtifactLoaderError(f"Failed to parse meta file {meta_path}: {exc}") from exc

    missing = [k for k in _REQUIRED_META_KEYS if k not in meta]
    if missing:
        raise ArtifactLoaderError(f"Meta file missing required keys: {missing}")

    if meta["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactLoaderError(
            f"Artifact schema version mismatch: expected {ARTIFACT_SCHEMA_VERSION}, "
            f"got {meta['artifact_schema_version']}"
        )

    return meta


def _parse_uuid(value: str) -> UUID:
    try:
        return UUID(str(value))
    except Exception as exc:
        raise ArtifactLoaderError(f"Invalid artifact_id in meta file: {value!r}") from exc


def _parse_datetime(value: str, field_name: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except Exception as exc:
        raise ArtifactLoaderError(f"Invalid {field_name} in meta file: {value!r}") from exc


def _parse_pred_len(value) -> int:
    try:
        pred_len = int(value)
    except Exception as exc:
        raise ArtifactLoaderError(f"Invalid pred_len in meta file: {value!r}") from exc
    if pred_len <= 0:
        raise ArtifactLoaderError(f"pred_len must be positive, got {pred_len}")
    return pred_len


def _parse_timeframe(value) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactLoaderError(f"Invalid timeframe in meta file: {value!r}")
    return value


def _load_dataframe(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise ArtifactLoaderError(f"Parquet file not found: {parquet_path}")

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        raise ArtifactLoaderError(f"Failed to read parquet file {parquet_path}: {exc}") from exc

    missing = [c for c in _REQUIRED_PARQUET_COLUMNS if c not in df.columns]
    if missing:
        raise ArtifactLoaderError(f"Parquet file missing required columns: {missing}")

    if df.empty:
        raise ArtifactLoaderError(f"Parquet file has no rows: {parquet_path}")

    return df


def _check_artifact_id_consistency(df: pd.DataFrame, expected_artifact_id: str) -> None:
    """
    Detects a crash-window mismatch between parquet and meta.json (e.g. a new
    parquet committed but the old meta.json survives). Every row's
    artifact_id must match the meta file's artifact_id exactly.
    """
    mismatched = df.loc[df["artifact_id"] != expected_artifact_id, "artifact_id"].unique()
    if len(mismatched) > 0:
        raise ArtifactLoaderError(
            f"artifact_id mismatch between parquet and meta.json: "
            f"expected {expected_artifact_id!r}, found {list(mismatched)!r} in parquet rows"
        )


def _build_record(symbol: str, group: pd.DataFrame) -> PredictionRecord:
    ranking_positions = group["ranking_position"].unique()
    ranking_scores = group["ranking_score"].unique()

    if len(ranking_positions) != 1:
        raise ArtifactLoaderError(
            f"Inconsistent ranking_position within symbol {symbol!r}: {list(ranking_positions)}"
        )
    if len(ranking_scores) != 1:
        raise ArtifactLoaderError(
            f"Inconsistent ranking_score within symbol {symbol!r}: {list(ranking_scores)}"
        )

    group = group.sort_values("timestamp")

    try:
        bars = tuple(
            PredictedBar(
                timestamp=row.timestamp.to_pydatetime()
                if hasattr(row.timestamp, "to_pydatetime")
                else row.timestamp,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                amount=float(row.amount),
            )
            for row in group.itertuples(index=False)
        )
    except Exception as exc:
        raise ArtifactLoaderError(f"Failed to reconstruct bars for symbol {symbol!r}: {exc}") from exc

    return PredictionRecord(
        symbol=symbol,
        forecast=ForecastSeries(bars=bars),
        ranking_position=int(ranking_positions[0]),
        ranking_score=float(ranking_scores[0]),
    )


def load_artifact(parquet_path: Path, meta_path: Path) -> PredictionArtifact:
    """
    Loads, validates, and reconstructs one PredictionArtifact from the
    parquet + meta.json pair produced by
    prediction_agent/artifact_builder.py.save_artifact().

    Validates (in order):
        - meta.json presence, parseability, required keys
        - artifact_schema_version match
        - artifact_id / datetime fields parseable
        - parquet presence, parseability, required columns, non-empty
        - artifact_id consistency between parquet rows and meta.json
        - artifact expiry (valid_until vs. now)
        - ranking_position / ranking_score consistency within each symbol

    Raises:
        ArtifactLoaderError: on any validation failure.
    """
    parquet_path = Path(parquet_path)
    meta_path = Path(meta_path)

    meta = _load_meta(meta_path)
    artifact_id = _parse_uuid(meta["artifact_id"])
    generated_at = _parse_datetime(meta["generated_at"], "generated_at")
    valid_from = _parse_datetime(meta["valid_from"], "valid_from")
    valid_until = _parse_datetime(meta["valid_until"], "valid_until")
    engine_reference = meta["engine_reference"]
    pred_len = _parse_pred_len(meta["pred_len"])
    timeframe = _parse_timeframe(meta["timeframe"])

    now = datetime.utcnow()

    df = _load_dataframe(parquet_path)
    _check_artifact_id_consistency(df, meta["artifact_id"])

    if valid_until <= now:
        raise ArtifactLoaderError(
            f"Artifact {artifact_id} expired: valid_until={valid_until.isoformat()}, now={now.isoformat()}"
        )

    records = [
        _build_record(symbol, group)
        for symbol, group in df.groupby("symbol", sort=False)
    ]
    records.sort(key=lambda r: r.ranking_position)

    artifact = PredictionArtifact(
        artifact_id=artifact_id,
        generated_at=generated_at,
        valid_from=valid_from,
        valid_until=valid_until,
        engine_reference=engine_reference,
        pred_len=pred_len,
        timeframe=timeframe,
        records=tuple(records),
    )

    logger.info(
        "Artifact %s loaded: %d records, valid until %s",
        artifact.artifact_id, len(records), artifact.valid_until,
    )
    return artifact