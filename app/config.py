from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "dev"
    log_level: str = "INFO"
    store_path: str = "data/store.db"
    model_path: str = "data/model.json"
    readmission_model_path: str = "data/readmission_model.json"
    storage_backend: str = "sqlite"
    database_url: str = "postgresql://medintel:medintel@postgres:5432/medintel"
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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("model_signature_required", mode="before")
    @classmethod
    def _empty_signature_is_unset(cls, value: object) -> object:
        # compose/.env 常用空串表达"未设置"；pydantic 不接受 "" 作为 bool。
        return None if value == "" else value

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
