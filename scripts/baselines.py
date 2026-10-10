"""NEWS2 early-warning baseline (T5.4).

RCP NEWS2 Scale 1：五个生理组件按区间计 0–3 分；意识项（ACVPU）演示数据
缺测，不计入总分——报告与文档注明该局限（Plan Ruling R9）。
"""

import math
from collections.abc import Mapping
from typing import Any


def _points(value: float, bands: tuple[tuple[float, int], ...]) -> int:
    """bands: ascending (upper_bound_inclusive, points); entry (-1, p) = +inf."""
    for upper, points in bands:
        if upper < 0 or value <= upper:
            return points
    raise ValueError("value outside NEWS2 band table")


RR_BANDS: tuple[tuple[float, int], ...] = ((8, 3), (11, 1), (20, 0), (24, 2), (-1, 3))
SPO2_BANDS: tuple[tuple[float, int], ...] = ((91, 3), (93, 2), (95, 1), (-1, 0))
SBP_BANDS: tuple[tuple[float, int], ...] = ((90, 3), (100, 2), (110, 1), (219, 0), (-1, 3))
HR_BANDS: tuple[tuple[float, int], ...] = ((40, 3), (50, 1), (90, 0), (110, 1), (130, 2), (-1, 3))
TEMP_BANDS: tuple[tuple[float, int], ...] = ((35.0, 3), (36.0, 1), (38.0, 0), (39.0, 1), (-1, 2))

COMPONENTS: tuple[tuple[str, tuple[tuple[float, int], ...]], ...] = (
    ("respiratory_rate", RR_BANDS),
    ("spo2", SPO2_BANDS),
    ("systolic_bp", SBP_BANDS),
    ("heart_rate", HR_BANDS),
    ("temperature", TEMP_BANDS),
)


def news2_score(row: Mapping[str, Any]) -> tuple[int, str]:
    """Return (total score 0-15, band). Missing components are skipped."""
    total = 0
    highest_single = 0
    any_component = False
    for key, bands in COMPONENTS:
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        points = _points(float(raw), bands)
        total += points
        highest_single = max(highest_single, points)
        any_component = True
    if not any_component:
        return 0, "low"
    # 官方升级规则：任一单参数 3 分 → high；否则按总分 0-4 low / 5-6 medium / >=7 high。
    if highest_single >= 3 or total >= 7:
        band = "high"
    elif total >= 5:
        band = "medium"
    else:
        band = "low"
    return total, band


# LACE 分值表（Austin et al., CMAJ 2010；Ruling R-P8-5：离线环境未能在线核对原文，
# 合并前请对照原文复核分箱边界）。总分 0–19。


def lace_score(row: Mapping[str, Any]) -> int:
    """Return the LACE readmission-risk index (0-19) for a discharge snapshot row."""
    required = ("los_days", "prior_utilization_12m", "comorbidity_index", "ed_visits_6m")
    values: dict[str, float] = {}
    for name in required:
        if name not in row or row[name] in ("", None):
            raise ValueError(f"lace fields missing/invalid: {name}")
        try:
            numeric = float(row[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"lace fields missing/invalid: {name}") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"lace fields missing/invalid: {name}")
        values[name] = numeric

    los = values["los_days"]
    # 边界以测试点为准（1.5 与 3.9 同落 2-3 档；台账 Task 3 Ruling）。
    length = 0 if los <= 1 else 1 if los < 4 else 2 if los < 7 else 3
    admissions = values["prior_utilization_12m"]
    admission = 0 if admissions < 1 else 2 if admissions < 2 else 4 if admissions < 3 else 7
    charlson = values["comorbidity_index"]
    comorbidity = 0 if charlson < 1 else 2 if charlson < 2 else 4 if charlson < 4 else 6
    visits = values["ed_visits_6m"]
    emergency = 0 if visits < 1 else 1 if visits < 2 else 2 if visits < 3 else 3
    return length + admission + comorbidity + emergency
