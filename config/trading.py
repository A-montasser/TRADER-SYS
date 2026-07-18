"""
config/trading.py

Trading Runtime Parameters — the single file an operator edits
day-to-day to tune how the system trades. Plain data, no logic, no
env-var indirection: open this file in VS Code, change a number,
save, run `python trading_bot.py`.

Contains ONLY parameters that:
    (a) affect trading/prediction behavior, and
    (b) have a real, existing consumer in the repository today.

Does NOT contain secrets or machine-specific settings (API keys,
exchange id, filesystem paths, log level, torch device) — those stay
in `.env` at the repository root; see config/config.py's module
docstring for the split rationale.

IMPORTANT — not every "strategy knob" you might expect lives here.
This project's decision_engine.py/risk_manager.py deliberately use
forecast-derived, per-opportunity exit thresholds (each Opportunity's
own drawdown_estimate_pct/upside_estimate_pct and bars_to_
profitability) rather than a single fixed profit/loss/hold-time
target. There is currently no global "profit threshold," "loss
threshold," or "hold-time limit" anywhere in the codebase to
configure — introducing one would mean changing decision_engine.py's
business logic itself (out of scope for the configuration layer; a
frozen-stage design decision, not a missing config value). If a
global-override exit policy is ever wanted, that is a decision_engine.py
architecture change to propose separately, not a config-layer addition.

config/config.py is the only module that reads this file — nothing
else should import config.trading directly, keeping "no other module
should know whether a value came from .env or here" true throughout
the rest of the repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class TradingParameters:
    """
    Every field has a literal default so the system can start with no
    edits at all — but INITIAL_BALANCE and the Kronos repo ids are real
    capital / real model selections and should always be reviewed
    before a live run, not trusted blindly.
    """

    # ── Trading Agent budget ────────────────────────────────────────
    # Starting quote-currency (e.g. USDT) balance TradingBot trades
    # with. Review this before every live run — it is real capital.
    initial_balance: float = 10.0

    # ── Shared OHLCV cadence ─────────────────────────────────────────
    # ccxt-style timeframe (e.g. "1m", "5m", "15m", "1h"). Feeds BOTH
    # the Prediction Agent's OHLCV downloader and Kronos's forecast
    # cadence — see prediction_agent/artifact_builder.py's
    # ArtifactBuilderConfig docstring: "not a separate concept."
    trading_timeframe: str = "5m"

    # ── Prediction Agent: market filtering ──────────────────────────
    allowed_quote_currencies: List[str] = field(default_factory=lambda: ["USDT"])
    # None = use prediction_agent/filters.py's own DEFAULT_HARAM_KEYWORDS.
    haram_keywords: Optional[List[str]] = None
    # None = skip this filter stage entirely (see filters.py).
    min_quote_volume_24h: Optional[float] = None
    max_spread_pct: Optional[float] = None

    # ── Prediction Agent: ranking ────────────────────────────────────
    # How many top-ranked symbols proceed to download + Kronos inference.
    ranking_top_n: int = 20
    # Optional momentum-strategy overrides — None means "use
    # prediction_agent/ranking.py's own RankingConfig default" for
    # that specific field.
    momentum_timeframe: Optional[str] = None
    momentum_limit: Optional[int] = None
    momentum_threshold_pct: Optional[float] = None
    momentum_up_factor: Optional[float] = None
    momentum_down_factor: Optional[float] = None
    momentum_neutral_factor: Optional[float] = None

    # ── Prediction Agent: downloader ─────────────────────────────────
    # Number of historical bars fetched per symbol.
    download_limit: int = 200

    # ── Prediction Agent: validator ──────────────────────────────────
    # None = use validator.py's own default (10).
    validator_min_bars: Optional[int] = None

    # ── Prediction Agent: Kronos ─────────────────────────────────────
    # Public HuggingFace repo ids — review before a live run just like
    # initial_balance; which model is loaded is a real strategy choice.
    kronos_tokenizer_repo_id: str = "NeoQuasar/Kronos-Tokenizer-base"
    kronos_model_repo_id: str = "NeoQuasar/Kronos-small"
    # Number of future bars Kronos forecasts per Prediction Cycle.
    kronos_pred_len: int = 24
    kronos_max_context: int = 512
    kronos_clip: int = 5
    kronos_temperature: float = 1.0
    kronos_top_k: int = 0
    kronos_top_p: float = 0.9
    kronos_sample_count: int = 1
    kronos_verbose: bool = False

    # ── Prediction Agent: artifact validity ──────────────────────────
    # How long a Prediction Artifact (Entry Forecast) remains valid
    # after generation, in minutes.
    artifact_validity_minutes: int = 30


# The single instance config/config.py loads. Edit the values above,
# not this line.
TRADING_PARAMETERS = TradingParameters()