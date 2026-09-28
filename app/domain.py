import math
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


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
    facility_id: str = "hospital-01"
    occurred_at: datetime = Field(default_factory=utc_now)
    detail: dict[str, Any] = Field(default_factory=dict)
