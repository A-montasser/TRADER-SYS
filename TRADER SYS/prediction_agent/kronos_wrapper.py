"""
prediction_agent/kronos_wrapper.py

Kronos Integration module for the Prediction Agent.

Responsibility (frozen):
    Pure translation boundary between ValidatedSeries and the isolated
    Kronos runtime (project/kronos). Converts validated bars into the
    DataFrame format Kronos requires, builds x_timestamp/y_timestamp
    from the configured timeframe, calls
    KronosPredictor.predict_batch(), and translates results into
    ForecastResult.

Explicitly NOT this module's responsibility:
    - Ranking                        -> prediction_agent/ranking.py
    - PredictionRecord construction   -> prediction_agent/artifact_builder.py
    - PredictionArtifact construction -> prediction_agent/artifact_builder.py
    - Forecast analytics              -> prediction_agent/analytics.py
    - Trading decisions               -> trading/
    - Retry policy                    -> prediction_agent/runtime.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from kronos import Kronos, KronosTokenizer, KronosPredictor
from models.market import ValidatedSeries
from models.artifact import ForecastSeries, ForecastResult, PredictedBar

logger = logging.getLogger(__name__)


class KronosWrapperError(Exception):
    """Raised when Kronos loading or inference fails."""


_TIMEFRAME_UNITS = {"m": "minutes", "h": "hours", "d": "days"}


def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """
    Converts a ccxt-style timeframe string (e.g. "1m", "5m", "1h") into
    a pd.Timedelta. Same convention already used by DownloaderConfig.
    """
    unit_char = timeframe[-1]
    if unit_char not in _TIMEFRAME_UNITS:
        raise KronosWrapperError(f"Unsupported timeframe unit: {timeframe}")
    value = int(timeframe[:-1])
    return pd.Timedelta(**{_TIMEFRAME_UNITS[unit_char]: value})


@dataclass(frozen=True)
class KronosWrapperConfig:
    """
    Mirrors prediction_agent.yaml Kronos parameters. Module-local,
    single-consumer — same established pattern as DownloaderConfig,
    ValidatorConfig, RankingConfig, FilterConfig.

    Loading sequence confirmed against the official Kronos
    prediction_example.py:
    
    KronosTokenizer.from_pretrained(tokenizer_repo_id)
    
    Kronos.from_pretrained(model_repo_id)
    
    KronosPredictor(model, tokenizer, ...)
    """
    tokenizer_repo_id: str
    model_repo_id: str
    pred_len: int
    timeframe: str
    max_context: int = 512
    clip: int = 5
    T: float = 1.0
    top_k: int = 0
    top_p: float = 0.9
    sample_count: int = 1
    verbose: bool = False
    device: Optional[str] = None


class KronosWrapper:
    """
    Owns the Kronos runtime (KronosPredictor) for the Prediction Agent.
    Constructed once per prediction_agent runtime lifecycle by runtime.py.
    """

    def __init__(self, config: KronosWrapperConfig):
        try:
            tokenizer = KronosTokenizer.from_pretrained(config.tokenizer_repo_id)
            model = Kronos.from_pretrained(config.model_repo_id)
        except Exception as exc:
            raise KronosWrapperError(f"Failed to load Kronos model/tokenizer: {exc}") from exc

        try:
            self._predictor = KronosPredictor(
                model,
                tokenizer,
                device=config.device,
                max_context=config.max_context,
                clip=config.clip,
            )
        except Exception as exc:
            raise KronosWrapperError(f"Failed to initialize KronosPredictor: {exc}") from exc

        self._config = config
        self._interval = _timeframe_to_timedelta(config.timeframe)
        logger.info(
            "KronosWrapper initialized (model=%s, tokenizer=%s, device=%s, timeframe=%s)",
            config.model_repo_id, config.tokenizer_repo_id,
            self._predictor.device, config.timeframe,
        )

    def generate_forecasts(
        self,
        validated_series: List[ValidatedSeries],
    ) -> tuple[ForecastResult, ...]:
        """
        Runs batch inference over all validated series.

        Args:
            validated_series: output of validator.py.

        Returns:
            tuple[ForecastResult, ...]: one per input series, in order.

        Raises:
            KronosWrapperError: if Kronos rejects the batch or inference
            otherwise fails.
        """
        if not validated_series:
            logger.info("KronosWrapper: no validated series to predict")
            return tuple()

        symbols: List[str] = []
        df_list: List[pd.DataFrame] = []
        # NOTE: these hold pd.Series, not pd.DatetimeIndex — see the
        # conversion below and its comment for why.
        x_timestamp_list: List[pd.Series] = []
        y_timestamp_list: List[pd.Series] = []

        for series in validated_series:
            symbol = series.symbol
            bars = series.bars

            # 'amount' intentionally omitted — Kronos derives it internally
            # from volume/price when absent; duplicating that derivation
            # here would violate the never-duplicate-normalization constraint.
            df = pd.DataFrame({
                "open": [b.open for b in bars],
                "high": [b.high for b in bars],
                "low": [b.low for b in bars],
                "close": [b.close for b in bars],
                "volume": [b.volume for b in bars],
            })

            x_timestamp = pd.DatetimeIndex(
                pd.Timestamp(b.timestamp_ms, unit="ms") for b in bars
            )
            y_timestamp = pd.DatetimeIndex(
                x_timestamp[-1] + self._interval * (i + 1)
                for i in range(self._config.pred_len)
            )

            symbols.append(symbol)
            df_list.append(df)
            # Kronos's own calc_time_stamps() (kronos/kronos.py) reads
            # x_timestamp.dt.minute/.dt.hour/etc. — the .dt accessor
            # only exists on pd.Series, not on pd.DatetimeIndex (a
            # DatetimeIndex exposes .minute/.hour directly, with no
            # .dt). predict_batch()'s own docstring says "DatetimeIndex
            # or Series" but its actual implementation only works with
            # Series; we build DatetimeIndex above because its
            # positional x_timestamp[-1] indexing is what
            # y_timestamp's own construction depends on (a bare Series
            # here would raise KeyError on [-1] against a default
            # RangeIndex, since Series.__getitem__ is label-based).
            # Converting to Series only at this translation boundary —
            # the moment these values cross into vendored Kronos code —
            # keeps the DatetimeIndex-based arithmetic above untouched
            # while satisfying what Kronos's implementation actually
            # requires. Values (and their order) are unaffected either
            # way; calc_time_stamps() only reads them positionally, and
            # Kronos's own predict_batch() uses the same Series
            # directly as pred_df's row index afterward — pandas
            # infers a proper DatetimeIndex from it there regardless.
            x_timestamp_list.append(pd.Series(x_timestamp))
            y_timestamp_list.append(pd.Series(y_timestamp))

        try:
            pred_dfs = self._predictor.predict_batch(
                df_list=df_list,
                x_timestamp_list=x_timestamp_list,
                y_timestamp_list=y_timestamp_list,
                pred_len=self._config.pred_len,
                T=self._config.T,
                top_k=self._config.top_k,
                top_p=self._config.top_p,
                sample_count=self._config.sample_count,
                verbose=self._config.verbose,
            )
        except Exception as exc:
            raise KronosWrapperError(f"Kronos batch inference failed: {exc}") from exc

        results: List[ForecastResult] = []
        for symbol, pred_df in zip(symbols, pred_dfs):
            bars_out = tuple(
                PredictedBar(
                    timestamp=ts.to_pydatetime(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    amount=float(row["amount"]),
                )
                for ts, row in pred_df.iterrows()
            )
            results.append(ForecastResult(symbol=symbol, forecast=ForecastSeries(bars=bars_out)))

        logger.info(
            "KronosWrapper: generated forecasts for %d/%d symbols",
            len(results), len(validated_series),
        )
        return tuple(results)