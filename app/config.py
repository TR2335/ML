import difflib
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DATABASE_URL = "postgresql://medintel:medintel@postgres:5432/medintel"


class Settings(BaseSettings):
    app_env: str = "dev"
    log_level: str = "INFO"
    store_path: str = "data/store.db"
    model_path: str = "data/model.json"
    readmission_model_path: str = "data/readmission_model.json"
    storage_backend: str = "sqlite"
    database_url: str = DEFAULT_DATABASE_URL
    event_backend: str = "sync"
    redis_url: str = "redis://redis:6379/0"
    redis_stream: str = "medintel:clinical-events"
    redis_dead_letter_stream: str = "medintel:clinical-events:dead-letter"
    redis_consumer_group: str = "medintel-workers"
    redis_consumer_name: str = "worker-1"
    redis_claim_idle_ms: int = 60000
    redis_max_delivery_attempts: int = 5
    metrics_enabled: bool = True
    model_signature_required: bool | None = None
    model_signature_path: str | None = None
    model_fallback_enabled: bool = False
    model_version: str = "baseline-logistic-v1"
    score_interval_seconds: int = 300
    reconcile_interval_cycles: int = 20
    max_event_future_skew_minutes: int = 15
    cors_origins: str = "http://localhost:8000"
    demo_auth_enabled: bool = True
    demo_clinician_token: str = "demo-clinician-token"
    demo_quality_token: str = "demo-quality-token"
    demo_admin_token: str = "demo-admin-token"
    oidc_enabled: bool = False
    oidc_issuer: str = ""
    oidc_audience: str = ""
    oidc_jwks_url: str = ""
    oidc_role_claim: str = "roles"
    oidc_tenant_claim: str = "tenant_id"
    oidc_facility_claim: str = "facility_id"
    mlflow_tracking_uri: str = ""  # 供外部工具读取；保住 .env.example 的合法条目（Ruling R3）
    postgres_password: str = ""  # compose 插值专用；应用不消费，放 .env 不得炸 strict 模式

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("model_signature_required", mode="before")
    @classmethod
    def _empty_signature_is_unset(cls, value: object) -> object:
        # compose/.env 常用空串表达"未设置"；pydantic 不接受 "" 作为 bool。
        return None if value == "" else value

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


class StrictSettings(Settings):
    """Non-dev parsing: unknown .env entries are rejected (Plan Ruling R2:
    OS environment variables have no prefix and cannot be detected)."""

    model_config = SettingsConfigDict(env_file=".env", extra="forbid")


def _detect_env_file_key_typos() -> None:
    """Dev path guard: a typoed selector key (e.g. APP_ENB) would otherwise
    silently fall back to dev and disable every fail-closed check (T3.5)."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    known = [name.upper() for name in Settings.model_fields]
    typos: list[tuple[str, str]] = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip().upper()
        if key in known:
            continue
        close = difflib.get_close_matches(key, known, n=1, cutoff=0.8)
        if close:
            typos.append((key, close[0]))
    if typos:
        detail = ", ".join(f"{key}≈{field}" for key, field in typos)
        raise RuntimeError(f"unknown .env keys look like typos of known settings: {detail}")


def build_settings() -> Settings:
    probe = Settings()
    if probe.app_env == "dev":
        _detect_env_file_key_typos()
        return probe
    return StrictSettings()


@lru_cache
def get_settings() -> Settings:
    return build_settings()


def validate_production_configuration(settings: Settings) -> None:
    if settings.app_env == "dev":
        return
    if settings.storage_backend == "postgres" and settings.database_url == DEFAULT_DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL must be set to a site-specific value outside dev "
            "(refusing the bundled default credentials)"
        )
