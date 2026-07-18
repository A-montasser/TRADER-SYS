"""
config/config.py

Deployment Configuration System — the integration layer between two
deliberately separate sources and the typed configuration objects
every other module already defines:

    .env (repository root, next to trading_bot.py)
        Secrets and machine/deployment-specific settings only:
        BINANCE_API_KEY, BINANCE_API_SECRET, EXCHANGE_ID, DATA_DIR and
        path overrides, LOG_LEVEL, KRONOS_DEVICE. Gitignored. An
        operator running this system day-to-day should almost never
        need to open it.

    config/trading.py
        Trading/prediction strategy parameters an operator is expected
        to tune routinely: initial_balance, timeframe, ranking top_n,
        download limit, artifact validity, Kronos inference settings,
        filter settings. Plain typed Python, not env-var indirection —
        this is the file meant to be opened and edited directly in
        VS Code before a run. Not a secret; normally version-controlled.

This module:
    - loads .env (via python-dotenv, if present)
    - imports config.trading.TRADING_PARAMETERS
    - validates both
    - builds the single DeploymentConfig below

Every other module in the repository consumes ONLY DeploymentConfig —
nothing downstream of load_config() knows or needs to know whether a
given value originated from .env or from config/trading.py. The
configuration objects produced (FilterConfig, RankingConfig,
DownloaderConfig, ValidatorConfig, ArtifactBuilderConfig,
KronosWrapperConfig, TradingBotConfig, Exchange credentials) are all
pre-existing shapes this module supplies values for — no new
configuration shape is introduced here or anywhere else.

Responsibility (this module only):
    - Read .env / real env vars (secrets, machine settings).
    - Read config/trading.py (strategy parameters).
    - Validate both. Fail fast with every problem found, not just the
      first.
    - Create missing runtime directories.
    - Validate exchange connectivity (credentials actually authenticate).
    - Configure logging.

Explicitly NOT this module's responsibility:
    - Any trading decision, risk computation, or prediction logic.
    - Constructing Exchange/KronosWrapper/TradingBot instances — that
      remains trading_bot.py's job (the project's sole entry point);
      this module only supplies the values those constructors need.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

from exchange import Exchange, ExchangeError
from prediction_agent.filters import DEFAULT_HARAM_KEYWORDS, FilterConfig
from prediction_agent.ranking import RankingConfig
from prediction_agent.downloader import DownloaderConfig
from prediction_agent.validator import ValidatorConfig
from prediction_agent.artifact_builder import ArtifactBuilderConfig
from prediction_agent.kronos_wrapper import KronosWrapperConfig

from config.trading import TRADING_PARAMETERS, TradingParameters

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """
    Raised when .env / config/trading.py cannot together produce a
    valid DeploymentConfig, when required runtime directories cannot
    be created, or when exchange connectivity cannot be confirmed.
    Always carries every problem found, not just the first — see
    load_config().
    """


@dataclass(frozen=True)
class DeploymentConfig:
    """
    Everything trading_bot.py's __main__ needs to construct Exchange,
    KronosWrapper, and TradingBotConfig. Plain data — no behavior, no
    mutable global state; load_config() returns a fresh instance per
    call. Deliberately does not distinguish which field came from .env
    vs. config/trading.py — that distinction ends here.
    """
    # Exchange credentials (exchange.py) — from .env
    exchange_id: str
    binance_api_key: str
    binance_api_secret: str

    # Trading Agent budget (trading_bot.py's TradingBot.initial_balance)
    # — from config/trading.py
    initial_balance: float

    # Prediction Agent sub-configs — from config/trading.py
    filter_config: FilterConfig
    ranking_config: RankingConfig
    downloader_config: DownloaderConfig
    validator_config: ValidatorConfig
    artifact_builder_config: ArtifactBuilderConfig
    kronos_wrapper_config: KronosWrapperConfig
    scanner_config: Dict[str, Any]

    # Runtime paths (trading_bot.py's TradingBotConfig) — from .env
    data_dir: Path
    artifacts_dir: Path
    trade_history_path: Path
    position_state_path: Path

    # Logging — from .env
    log_file: Path
    log_level: str


class _ErrorCollector:
    """
    Accumulates every problem found across BOTH .env and
    config/trading.py instead of raising on the first one, so
    load_config() fails fast with one complete, human-readable list —
    not a frustrating one-at-a-time trial-and-error loop.
    """

    def __init__(self) -> None:
        self.errors: List[str] = []

    def add(self, message: str) -> None:
        self.errors.append(message)


class _EnvReader:
    """
    Reads .env / real env vars — secrets and machine-specific settings
    ONLY (BINANCE_API_KEY, BINANCE_API_SECRET, EXCHANGE_ID, DATA_DIR
    and path overrides, LOG_LEVEL, KRONOS_DEVICE). Strategy parameters
    live in config/trading.py instead — see _TradingParamsReader below.
    """

    def __init__(self, errors: _ErrorCollector) -> None:
        self._errors = errors

    def require_str(self, name: str) -> str:
        value = os.environ.get(name, "").strip()
        if not value:
            self._errors.add(f"{name} is required in .env and was not set (or is empty)")
            return ""
        return value

    def optional_str(self, name: str, default: str) -> str:
        return os.environ.get(name, default).strip() or default


def _load_dotenv_if_present() -> None:
    """
    Loads a `.env` file from the current working directory (repository
    root, next to trading_bot.py — NOT inside config/) if present,
    without overriding variables already set in the real environment
    (so a deployment that sets real env vars — e.g. a container/systemd
    setup — is never silently shadowed by a stray local .env file).
    Never fails startup on its own — an absent .env is expected in some
    deployments (env vars set directly), and is not an error by itself;
    load_config()'s own validation catches genuinely missing values.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning(
            "python-dotenv is not installed — .env file (if any) will not be "
            "loaded automatically; relying on real environment variables only. "
            "Install with: pip install python-dotenv"
        )
        return
    load_dotenv(override=False)


def _validate_trading_parameters(params: TradingParameters, errors: _ErrorCollector) -> None:
    """
    Bounds/consistency validation for config/trading.py's values.
    Unlike _EnvReader, these arrive already correctly typed (real
    float/int/str, not strings to parse) since config/trading.py is a
    plain Python module, not env-var text — this only checks value
    ranges, not types.
    """
    if params.initial_balance <= 0:
        errors.add(f"config/trading.py: initial_balance={params.initial_balance} must be positive")

    if not params.trading_timeframe:
        errors.add("config/trading.py: trading_timeframe must not be empty")

    if not params.allowed_quote_currencies:
        errors.add("config/trading.py: allowed_quote_currencies must not be empty")

    if params.ranking_top_n <= 0:
        errors.add(f"config/trading.py: ranking_top_n={params.ranking_top_n} must be positive")

    if params.download_limit <= 0:
        errors.add(f"config/trading.py: download_limit={params.download_limit} must be positive")

    if params.validator_min_bars is not None and params.validator_min_bars <= 0:
        errors.add(
            f"config/trading.py: validator_min_bars={params.validator_min_bars} "
            "must be positive (or None to use the repository default)"
        )

    if not params.kronos_tokenizer_repo_id:
        errors.add("config/trading.py: kronos_tokenizer_repo_id must not be empty")
    if not params.kronos_model_repo_id:
        errors.add("config/trading.py: kronos_model_repo_id must not be empty")
    if params.kronos_pred_len <= 0:
        errors.add(f"config/trading.py: kronos_pred_len={params.kronos_pred_len} must be positive")
    if params.kronos_max_context <= 0:
        errors.add(f"config/trading.py: kronos_max_context={params.kronos_max_context} must be positive")
    if params.kronos_sample_count <= 0:
        errors.add(f"config/trading.py: kronos_sample_count={params.kronos_sample_count} must be positive")

    if params.artifact_validity_minutes <= 0:
        errors.add(
            f"config/trading.py: artifact_validity_minutes={params.artifact_validity_minutes} "
            "must be positive"
        )


def load_config() -> DeploymentConfig:
    """
    Reads and validates the full deployment configuration from BOTH
    .env (secrets/machine settings) and config/trading.py (strategy
    parameters). Raises ConfigError listing every problem found across
    both sources if anything is missing or invalid; never returns a
    partially-valid config.
    """
    _load_dotenv_if_present()
    errors = _ErrorCollector()
    env = _EnvReader(errors)
    params = TRADING_PARAMETERS
    _validate_trading_parameters(params, errors)

    # ── .env: exchange credentials / machine settings ────────────────
    exchange_id = env.optional_str("EXCHANGE_ID", "binance")
    binance_api_key = env.require_str("BINANCE_API_KEY")
    binance_api_secret = env.require_str("BINANCE_API_SECRET")

    # ── config/trading.py: Prediction Agent filters ──────────────────
    allowed_quotes = [q.strip().upper() for q in params.allowed_quote_currencies if q.strip()]
    haram_keywords = (
        list(params.haram_keywords) if params.haram_keywords is not None
        else list(DEFAULT_HARAM_KEYWORDS)
    )
    filter_config = FilterConfig(
        allowed_quote_currencies=allowed_quotes,
        haram_keywords=haram_keywords,
        min_quote_volume_24h=params.min_quote_volume_24h,
        max_spread_pct=params.max_spread_pct,
    )

    # ── config/trading.py: ranking (only override fields the operator
    #    actually set — None means "use ranking.py's own default") ───
    ranking_kwargs: Dict[str, Any] = {"top_n": params.ranking_top_n}
    for field_name in (
        "momentum_timeframe", "momentum_limit", "momentum_threshold_pct",
        "momentum_up_factor", "momentum_down_factor", "momentum_neutral_factor",
    ):
        value = getattr(params, field_name)
        if value is not None:
            ranking_kwargs[field_name] = value
    ranking_config = RankingConfig(**ranking_kwargs)

    # ── config/trading.py: downloader (shares trading_timeframe — see
    #    ArtifactBuilderConfig's own docstring: "not a separate
    #    concept," same principle applied here) ───────────────────────
    downloader_config = DownloaderConfig(
        timeframe=params.trading_timeframe,
        limit=params.download_limit,
    )

    # ── config/trading.py: validator ──────────────────────────────────
    validator_kwargs: Dict[str, Any] = {}
    if params.validator_min_bars is not None:
        validator_kwargs["min_bars"] = params.validator_min_bars
    validator_config = ValidatorConfig(**validator_kwargs)

    # ── config/trading.py: Kronos ──────────────────────────────────────
    kronos_wrapper_config = KronosWrapperConfig(
        tokenizer_repo_id=params.kronos_tokenizer_repo_id,
        model_repo_id=params.kronos_model_repo_id,
        pred_len=params.kronos_pred_len,
        timeframe=params.trading_timeframe,
        max_context=params.kronos_max_context,
        clip=params.kronos_clip,
        T=params.kronos_temperature,
        top_k=params.kronos_top_k,
        top_p=params.kronos_top_p,
        sample_count=params.kronos_sample_count,
        verbose=params.kronos_verbose,
        # KRONOS_DEVICE is machine-specific (which GPU/CPU is physically
        # present), not a strategy choice — stays in .env.
        device=env.optional_str("KRONOS_DEVICE", "") or None,
    )

    # engine_reference is fully determined by which model is loaded —
    # no separate value needed for audit metadata already implied by
    # kronos_model_repo_id.
    artifact_builder_config = ArtifactBuilderConfig(
        validity_window=timedelta(minutes=params.artifact_validity_minutes),
        engine_reference=f"kronos:{kronos_wrapper_config.model_repo_id}",
        pred_len=kronos_wrapper_config.pred_len,
        timeframe=params.trading_timeframe,
    )

    # scanner.py's config param is currently unused (documented in
    # scanner.py itself — no scanner-specific parameters exist yet);
    # not sourcing anything for it would be inventing config values
    # that have nowhere to go.
    scanner_config: Dict[str, Any] = {}

    # ── .env: runtime paths (machine-specific filesystem locations) ──
    data_dir = Path(env.optional_str("DATA_DIR", "./data"))
    artifacts_dir = Path(env.optional_str("ARTIFACTS_DIR", str(data_dir / "artifacts")))
    trade_history_path = Path(
        env.optional_str("TRADE_HISTORY_PATH", str(data_dir / "journal" / "trade_history.csv"))
    )
    position_state_path = Path(
        env.optional_str("POSITION_STATE_PATH", str(data_dir / "state" / "position_state.json"))
    )
    log_file = Path(env.optional_str("LOG_FILE", str(data_dir / "logs" / "trading_bot.log")))
    log_level = env.optional_str("LOG_LEVEL", "INFO").upper()
    if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        errors.add(
            f".env: LOG_LEVEL={log_level!r} is not a valid logging level "
            "(DEBUG, INFO, WARNING, ERROR, CRITICAL)"
        )

    if errors.errors:
        message = (
            f"Configuration invalid — {len(errors.errors)} problem(s) found "
            "across .env and config/trading.py:\n"
            + "\n".join(f"  - {e}" for e in errors.errors)
        )
        raise ConfigError(message)

    return DeploymentConfig(
        exchange_id=exchange_id,
        binance_api_key=binance_api_key,
        binance_api_secret=binance_api_secret,
        initial_balance=params.initial_balance,
        filter_config=filter_config,
        ranking_config=ranking_config,
        downloader_config=downloader_config,
        validator_config=validator_config,
        artifact_builder_config=artifact_builder_config,
        kronos_wrapper_config=kronos_wrapper_config,
        scanner_config=scanner_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        trade_history_path=trade_history_path,
        position_state_path=position_state_path,
        log_file=log_file,
        log_level=log_level,
    )


def ensure_runtime_directories(config: DeploymentConfig) -> None:
    """
    Creates every runtime directory the system will write to, so a
    fresh deployment never fails partway through a cycle because a
    directory was never created manually. Idempotent — safe to call
    on every startup, not just the first.

    Raises:
        ConfigError: if a directory genuinely cannot be created
            (permissions, disk full, path collides with a file, etc.)
            — wrapped from the underlying OSError so callers only need
            to catch one exception type from this module.
    """
    directories = {
        config.data_dir,
        config.artifacts_dir,
        config.trade_history_path.parent,
        config.position_state_path.parent,
        config.log_file.parent,
    }
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(f"Could not create runtime directory {directory}: {exc}") from exc


def configure_logging(config: DeploymentConfig) -> None:
    """
    Configures the root logger once, at startup, from DeploymentConfig
    — every module's existing `logger = logging.getLogger(__name__)`
    picks this up automatically since none of them configure handlers
    themselves. Writes to both the configured log file and stdout, so
    a deployment tailing stdout (e.g. `docker logs`) and one relying on
    the log file both work without extra setup.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(config.log_level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    file_handler = logging.FileHandler(config.log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # ccxt's own logger is extremely verbose at INFO — same suppression
    # pattern already established elsewhere in this project's history.
    logging.getLogger("ccxt.base.exchange").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def validate_exchange_connectivity(exchange: Exchange) -> None:
    """
    Confirms the configured Binance credentials actually authenticate
    before TradingBot starts, rather than discovering an auth failure
    on the first order attempt mid-cycle. Delegates the actual ccxt
    call to Exchange.verify_credentials() — exchange.py remains the
    sole ccxt translation boundary; this function only translates a
    failure into the same ConfigError type as every other startup
    validation step, so trading_bot.py's __main__ only needs to catch
    one exception type across the whole startup sequence.

    Raises:
        ConfigError: if authentication fails for any reason.
    """
    try:
        exchange.verify_credentials()
    except ExchangeError as exc:
        raise ConfigError(
            f"Exchange authentication failed — check BINANCE_API_KEY / "
            f"BINANCE_API_SECRET in .env: {exc}"
        ) from exc