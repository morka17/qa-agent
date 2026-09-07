"""
Application settings for Sentinel-QA.

All configuration is environment-driven via pydantic-settings, so the same
codebase runs unmodified across local/dev/staging/prod — only the .env
(or real environment variables / secrets manager) changes.

Usage:
    from qa_agent.config.settings import get_settings

    settings = get_settings()
    settings.llm_provider  # "anthropic"
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LOCAL = "local"


class IssueTracker(str, Enum):
    JIRA = "jira"
    LINEAR = "linear"
    GITHUB = "github"
    NONE = "none"


class Settings(BaseSettings):
    """
    Central, validated configuration object.

    Every field can be overridden via an environment variable of the same
    name (case-insensitive), or via a `.env` file in the project root.
    Secrets use `SecretStr` so they never leak into logs or repr().
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Core ---
    app_name: str = "sentinel-qa"
    environment: Environment = Environment.LOCAL
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    debug: bool = False

    # --- LLM provider ---
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    llm_model: str = "claude-sonnet-4-6"
    llm_api_key: SecretStr = Field(default=SecretStr(""))
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.0
    llm_request_timeout_s: int = 60

    # --- Database ---
    database_url: str = "postgresql+asyncpg://sentinel:sentinel@localhost:5432/sentinel_qa"
    database_pool_size: int = 10
    database_echo_sql: bool = False

    # --- Redis / task queue ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Artifact storage (video, trace, screenshots) ---
    artifact_bucket: str = "sentinel-qa-artifacts"
    artifact_store_provider: Literal["s3", "gcs", "local"] = "local"
    artifact_local_path: str = "./runs"

    # --- Playwright / execution ---
    playwright_headless: bool = True
    playwright_browser: Literal["chromium", "firefox", "webkit"] = "chromium"
    playwright_default_timeout_ms: int = 15_000
    playwright_navigation_timeout_ms: int = 30_000
    max_concurrent_browser_contexts: int = 5

    # --- Issue tracker filing ---
    issue_tracker: IssueTracker = IssueTracker.NONE

    jira_base_url: AnyHttpUrl | None = None
    jira_email: str | None = None
    jira_api_token: SecretStr | None = None
    jira_project_key: str | None = None

    linear_api_key: SecretStr | None = None
    linear_team_id: str | None = None

    github_token: SecretStr | None = None
    github_repo: str | None = None  # "org/repo"

    # --- Ingestion connectors ---
    csv_default_delimiter: str = ","

    # --- Guardrails ---
    destructive_action_allowlist: list[str] = Field(default_factory=list)
    require_human_approval_for_destructive_actions: bool = True

    @field_validator("llm_temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("llm_temperature must be between 0.0 and 1.0")
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PROD


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached, process-wide Settings instance.

    Cached with lru_cache so environment parsing/validation happens once
    per process rather than on every access. Tests can bypass the cache
    via `get_settings.cache_clear()` after monkeypatching env vars.
    """
    return Settings()