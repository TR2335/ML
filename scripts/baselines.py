"""NEWS2 early-warning baseline (T5.4).

RCP NEWS2 Scale 1：五个生理组件按区间计 0–3 分；意识项（ACVPU）演示数据
缺测，不计入总分——报告与文档注明该局限（Plan Ruling R9）。
"""

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
