"""
config/

Deployment configuration package. Owns the boundary between the
process environment (env vars, .env file) and every other module's
typed configuration objects (FilterConfig, RankingConfig,
DownloaderConfig, ValidatorConfig, ArtifactBuilderConfig,
KronosWrapperConfig, TradingBotConfig, Exchange credentials).

Nothing in this package contains trading, prediction, or business
logic — it only reads the environment, validates it, and constructs
the same configuration dataclasses each module already defines and
consumes. No new configuration *shape* is introduced anywhere else in
the repository; this package only supplies values for shapes that
already existed.
"""

from config.config import (
    ConfigError,
    DeploymentConfig,
    configure_logging,
    ensure_runtime_directories,
    load_config,
    validate_exchange_connectivity,
)

__all__ = [
    "ConfigError",
    "DeploymentConfig",
    "configure_logging",
    "ensure_runtime_directories",
    "load_config",
    "validate_exchange_connectivity",
]