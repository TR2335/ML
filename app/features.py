"""Shared feature order for training and serving (T5.2+).

The seven-tuple expansion (latest/mean/slope/min/max/missing/since_last) is
the contract both the offline pipeline and future trainers consume; serving
keeps raw-feature scoring until artifacts carry their own feature_order.
"""

import json
import math
from collections.abc import Mapping
from datetime import datetime, timedelta

from app.domain import ClinicalEvent
from app.synthetic import FEATURES

STAT_SUFFIXES = ("latest", "mean", "slope", "min", "max", "missing", "since_last")

TUPLE_FEATURE_ORDER: list[str] = [
    f"{feature}__{suffix}" for feature in FEATURES for suffix in STAT_SUFFIXES
]

# T5.4：七元组之后追加 news2_score（stacking 基线输入；评估用增益决定去留）。
FEATURE_ORDER: list[str] = list(TUPLE_FEATURE_ORDER) + ["news2_score"]


def build_landmarks(
    patient_events: dict[str, list[ClinicalEvent]],
    index_times: list[tuple[str, datetime]],
    window_hours: int = 24,
) -> list[dict]:
    """Build seven-tuple features per (encounter, index) landmark.

    Window is half-open [index - window_hours, index): events at the index
    instant are future information and must never leak in (Plan Ruling R6).
    index_times is a list of (encounter_id, index_time) pairs (Plan Ruling R13).
    """
    window = timedelta(hours=window_hours)
    rows: list[dict] = []
    for encounter_id, index_time in sorted(index_times):
        window_events = sorted(
            (
                event
                for event in patient_events.get(encounter_id, [])
                if index_time - window <= event.occurred_at < index_time
            ),
            key=lambda event: event.occurred_at,
        )
        row: dict = {"encounter_id": encounter_id, "index_time": index_time.isoformat()}
        for feature in FEATURES:
            samples = [
                (event.occurred_at, float(event.payload[feature]))
                for event in window_events
                if event.payload.get(feature) is not None
            ]
            row.update(_feature_stats(feature, samples, index_time))
        rows.append(row)
    return rows


def _feature_stats(
    feature: str, samples: list[tuple[datetime, float]], index_time: datetime
) -> dict[str, float | None]:
    prefix = f"{feature}__"
    if not samples:
        return {
            prefix + "latest": None,
            prefix + "mean": None,
            prefix + "slope": None,
            prefix + "min": None,
            prefix + "max": None,
            prefix + "missing": 1.0,
            prefix + "since_last": None,
        }
    values = [value for _, value in samples]
    last_time = samples[-1][0]
    return {
        prefix + "latest": samples[-1][1],
        prefix + "mean": round(sum(values) / len(values), 6),
        prefix + "slope": _least_squares_slope(samples),
        prefix + "min": min(values),
        prefix + "max": max(values),
        prefix + "missing": 0.0,
        prefix + "since_last": round((index_time - last_time).total_seconds() / 3600, 4),
    }


def _least_squares_slope(samples: list[tuple[datetime, float]]) -> float | None:
    """Slope per hour with the first sample as origin; None when n < 3."""
    if len(samples) < 3:
        return None
    origin = samples[0][0]
    points = [((t - origin).total_seconds() / 3600, value) for t, value in samples]
    count = len(points)
    mean_t = sum(t for t, _ in points) / count
    mean_v = sum(v for _, v in points) / count
    denom = sum((t - mean_t) ** 2 for t, _ in points)
    if denom == 0:
        return None
    cov = sum((t - mean_t) * (v - mean_v) for t, v in points)
    return round(cov / denom, 6)


READMISSION_PAYLOAD_KEYS = (
    "los_days",
    "prior_utilization_12m",
    "comorbidity_index",
    "ed_visits_6m",
    "diagnosis_codes",
    "discharge_destination",
)
READMISSION_FEATURES = [
    "los_days",
    "prior_utilization_12m",
    "comorbidity_index",
    "ed_visits_6m",
    "diagnosis_code_count",
    "discharge_destination_code",
]
DISCHARGE_DESTINATION_CODES = {"home": 0.0, "home_health": 1.0, "snf": 2.0, "other": 3.0}


def encode_readmission_payload(payload: Mapping) -> list[float]:
    """出院快照 → 6 维再入院特征（训练/serving 唯一编码源，Ruling R-P8-3）。

    源字段缺失填 NaN（与 P6 中位数填补语义一致）。
    """

    def _number(name: str) -> float:
        raw = payload.get(name)
        if raw in ("", None):
            return math.nan
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return math.nan
        return value if math.isfinite(value) else math.nan

    codes = payload.get("diagnosis_codes")
    if isinstance(codes, str) and codes.strip():
        try:
            codes = json.loads(codes)
        except json.JSONDecodeError:
            codes = None
    count = float(len(codes)) if isinstance(codes, list) else math.nan
    destination = payload.get("discharge_destination")
    destination_code = DISCHARGE_DESTINATION_CODES.get(str(destination), math.nan)
    return [
        _number("los_days"),
        _number("prior_utilization_12m"),
        _number("comorbidity_index"),
        _number("ed_visits_6m"),
        count,
        destination_code,
    ]
