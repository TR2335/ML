import json
import math
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


CLINICAL_NUMERIC_FIELDS = {
    "age": (0, 120),
    "heart_rate": (20, 300),
    "respiratory_rate": (4, 100),
    "systolic_bp": (40, 300),
    "temperature": (25, 45),
    "spo2": (0, 100),
    "wbc": (0, 200),
    "creatinine": (0, 50),
    "lactate": (0, 50),
    "prior_admissions_30d": (0, 100),
}

TargetId = Literal["deterioration_24h", "readmission_30d"]

DISCHARGE_DESTINATIONS = ("home", "home_health", "snf", "other")
DISCHARGE_SNAPSHOT_FIELDS = (
    "discharge_destination",
    "los_days",
    "diagnosis_codes",
    "comorbidity_index",
    "prior_utilization_12m",
    "ed_visits_6m",
)
# 出院快照数值范围：名称 -> (下界, 上界)；diagnosis_codes/discharge_destination 单独校验。
DISCHARGE_NUMERIC_RANGES = {
    "los_days": (0.0, 365.0),
    "comorbidity_index": (0.0, 40.0),
    "prior_utilization_12m": (0, 100),
    "ed_visits_6m": (0, 100),
}


class PatientRecord(BaseModel):
    patient_id: str = Field(min_length=1, max_length=128)
    encounter_id: str = Field(min_length=1, max_length=128)
    display_name: str = Field(default="", max_length=256)
    age: int = Field(ge=0, le=120)
    sex: Literal["M", "F", "O", "U"] = "U"
    department: str = Field(default="unknown", min_length=1, max_length=128)
    admitted_at: datetime = Field(default_factory=utc_now)
    status: Literal["inpatient", "discharged", "cancelled"] = "inpatient"
    tenant_id: str = Field(default="group-demo", min_length=1, max_length=128)
    facility_id: str = Field(default="hospital-01", min_length=1, max_length=128)


class ClinicalEvent(BaseModel):
    event_id: str = Field(
        default_factory=lambda: f"evt-{uuid4().hex}", min_length=1, max_length=128
    )
    event_type: Literal["vital_signs", "lab_result", "admission", "discharge", "medication", "note"]
    occurred_at: datetime = Field(default_factory=utc_now)
    tenant_id: str = Field(default="group-demo", min_length=1, max_length=128)
    facility_id: str = Field(default="hospital-01", min_length=1, max_length=128)
    patient_id: str = Field(min_length=1, max_length=128)
    encounter_id: str = Field(min_length=1, max_length=128)
    actor: str = Field(default="system", min_length=1, max_length=128)
    source_system: str | None = Field(default=None, max_length=128)
    source_record_id: str | None = Field(default=None, max_length=256)
    unit: dict[str, str] = Field(default_factory=dict, max_length=100)
    quality_status: Literal["unknown", "valid", "suspect", "invalid"] = "unknown"
    payload: dict[str, Any] = Field(default_factory=dict, max_length=100)

    @field_validator("occurred_at", mode="before")
    @classmethod
    def ensure_timezone(cls, value: Any) -> datetime:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        # Naive source timestamps are interpreted as UTC at the ingestion boundary.
        normalized = parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        return normalized.astimezone(UTC)

    @field_validator("payload")
    @classmethod
    def validate_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False)
        if len(encoded.encode("utf-8")) > 65536:
            raise ValueError("payload exceeds 65536 bytes")
        for field, value in payload.items():
            if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > 4096:
                raise ValueError(f"{field} value exceeds 4096 bytes")
        for field, (minimum, maximum) in CLINICAL_NUMERIC_FIELDS.items():
            if field not in payload:
                continue
            value = payload[field]
            if isinstance(value, bool):
                # ValueError 让 pydantic 包装成 422；TypeError 会漏成 500（T3.3）。
                raise ValueError(f"{field} must be numeric")  # noqa: TRY004
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be numeric") from exc
            if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
                raise ValueError(f"{field} must be between {minimum} and {maximum}")
        return payload

    @model_validator(mode="after")
    def validate_discharge_snapshot(self) -> "ClinicalEvent":
        if self.event_type != "discharge":
            return self
        missing = [name for name in DISCHARGE_SNAPSHOT_FIELDS if name not in self.payload]
        if missing:
            raise ValueError("discharge payload missing fields: " + ", ".join(missing))
        destination = self.payload["discharge_destination"]
        if not isinstance(destination, str) or destination not in DISCHARGE_DESTINATIONS:
            raise ValueError(
                f"discharge_destination must be one of {', '.join(DISCHARGE_DESTINATIONS)}"
            )
        codes = self.payload["diagnosis_codes"]
        if (
            not isinstance(codes, list)
            or len(codes) > 50
            or not all(isinstance(code, str) and 0 < len(code) <= 16 for code in codes)
        ):
            raise ValueError("diagnosis_codes must be a list of up to 50 short strings")
        for name, (minimum, maximum) in DISCHARGE_NUMERIC_RANGES.items():
            value = self.payload[name]
            if isinstance(value, bool):
                # ValueError 让 pydantic 包装成 422；TypeError 会漏成 500（T3.3）。
                raise ValueError(f"{name} must be numeric")  # noqa: TRY004
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
        return self


def targets_for_event(event: ClinicalEvent) -> list[TargetId]:
    """评分触发策略：deterioration 逐事件；readmission 仅出院事件（WBS T8.2）。"""
    targets: list[TargetId] = ["deterioration_24h"]
    if event.event_type == "discharge":
        targets.append("readmission_30d")
    return targets


class RiskFactor(BaseModel):
    feature: str
    label: str
    contribution: float
    value: float | str | None = None


class RiskScore(BaseModel):
    score_id: str = Field(default_factory=lambda: f"score-{uuid4().hex[:10]}")
    patient_id: str
    encounter_id: str
    tenant_id: str = "group-demo"
    facility_id: str = "hospital-01"
    target_id: TargetId = "deterioration_24h"
    model_version: str
    score: float = Field(ge=0, le=1)
    band: Literal["low", "medium", "high", "critical"]
    top_factors: list[RiskFactor] = Field(default_factory=list)
    missing_features: list[str] = Field(default_factory=list)
    scored_at: datetime = Field(default_factory=utc_now)
    as_of: datetime | None = None
    status: Literal["new", "acknowledged", "in_progress", "resolved"] = "new"
    human_review: str | None = None


class ReviewRequest(BaseModel):
    target_id: TargetId = "deterioration_24h"
    status: Literal["acknowledged", "in_progress", "resolved"]
    note: str = Field(min_length=1, max_length=1000)


class AuditEvent(BaseModel):
    audit_id: str = Field(default_factory=lambda: f"audit-{uuid4().hex}")
    action: str
    actor: str
    resource_type: str
    resource_id: str
    tenant_id: str = "group-demo"
    facility_id: str = "hospital-01"  # 兼容字段：迁移期与 actor_facility_id 同值
    actor_facility_id: str = ""
    resource_facility_id: str = ""
    occurred_at: datetime = Field(default_factory=utc_now)
    detail: dict[str, Any] = Field(default_factory=dict)
