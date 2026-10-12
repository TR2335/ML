"""Batch drift monitoring against an artifact's stored reference (WBS T7.4)."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# 直接以 `python scripts/x.py` 运行时根目录不在 sys.path（Ruling R14）。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features import READMISSION_FEATURES, encode_readmission_payload

PSI_THRESHOLD = 0.2  # WBS T7.4
KS_THRESHOLD = 0.1  # G4：效应量阈值（不看 p 值——大样本 p 必然显著，R-RS-14）


def _finite(value) -> float | None:
    """样本值解析（与 compute_psi 同语义）：空/非数字/非有限 → None（跳过）。"""
    if value in ("", None):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def compute_ks(reference_sample: list | None, values: list) -> float | None:
    """两样本 KS 统计量 max|F₁−F₂|（纯排序双指针，不引入 scipy）。

    无参考样本/任一侧无有效值 → None（跳过不判失败）。工程上只看效应量、
    不纳入 p 值（样本量大时 p 必然显著——本轮裁定）。
    """
    if not reference_sample:
        return None
    left = sorted(v for v in (_finite(x) for x in reference_sample) if v is not None)
    right = sorted(v for v in (_finite(x) for x in values) if v is not None)
    if not left or not right:
        return None
    i = j = 0
    delta = 0.0
    n, m = len(left), len(right)
    while i < n and j < m:
        if left[i] < right[j]:
            i += 1
        elif left[i] > right[j]:
            j += 1
        else:
            # 相同值必须一次性步进完：逐元素部分步进会制造虚假的中间 CDF 状态
            value = left[i]
            while i < n and left[i] == value:
                i += 1
            while j < m and right[j] == value:
                j += 1
        delta = max(delta, abs(i / n - j / m))
    return max(delta, abs(i / n - j / m))


def drift_alert(psi: float, ks: float | None, ks_threshold: float = KS_THRESHOLD) -> bool:
    """双指标告警：psi > 0.2 OR ks > 阈值；ks 不可用（None）不阻断。"""
    return psi > PSI_THRESHOLD or (ks is not None and ks > ks_threshold)


def _feature_column(feature: str, rows: list[dict]) -> list:
    """特征列取值：输入含该列直取；readmission 派生特征缺列时按源字段即时编码
    （serving R2 同语义——漂移输入可以是出院快照源字段 CSV，R-RS-15）。"""
    if feature in rows[0]:
        return [row.get(feature, "") for row in rows]
    if feature in READMISSION_FEATURES:
        lookup = {name: index for index, name in enumerate(READMISSION_FEATURES)}
        column: list = []
        for row in rows:
            value = encode_readmission_payload(row)[lookup[feature]]
            column.append("" if math.isnan(value) else value)
        return column
    return [row.get(feature, "") for row in rows]


def compute_psi(reference: dict, values: list, bins: int = 10) -> float:
    """PSI over reference bins（Ruling R-P7-9）；缺失并入独立格；epsilon 防零。"""
    lo, hi = float(reference["lo"]), float(reference["hi"])
    expected = list(reference["proportions"])
    width = (hi - lo) / bins if hi > lo else 1.0
    counts = [0.0] * bins
    missing = 0
    total = max(len(values), 1)
    for value in values:
        if value in ("", None):
            missing += 1
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            missing += 1
            continue
        if not math.isfinite(numeric):
            missing += 1
            continue
        if numeric < lo:
            counts[0] += 1
        elif numeric > hi:
            counts[-1] += 1
        else:
            counts[min(int((numeric - lo) / width), bins - 1)] += 1
    actual = [c / total for c in counts]
    expected_missing = float(reference.get("missing_rate", 0.0))
    actual_missing = missing / total
    # 业界惯例 0.005 对称平滑：空箱不再被 ε=1e-6 级别的取对数放大（台账 Task 5）。
    smooth = 0.005
    psi = 0.0
    for a, e in zip(actual, expected):
        a2, e2 = a + smooth, e + smooth
        psi += (a2 - e2) * math.log(a2 / e2)
    am, em = actual_missing + smooth, expected_missing + smooth
    psi += (am - em) * math.log(am / em)
    return psi


def main() -> None:
    parser = argparse.ArgumentParser(description="PSI+KS drift check against artifact reference")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--input", required=True, help="CSV batch to compare")
    parser.add_argument("--ks-threshold", type=float, default=KS_THRESHOLD, help="KS 效应量阈值")
    args = parser.parse_args()
    artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    reference = artifact.get("drift_reference")
    if not isinstance(reference, dict) or not reference:
        raise SystemExit(f"artifact has no drift_reference: {args.artifact}")
    with Path(args.input).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("monitor_drift: empty input")
    report: dict = {"artifact": args.artifact, "rows": len(rows), "features": {}, "alert": False}
    for feature in artifact.get("feature_order", list(reference.keys())):
        if feature not in reference:
            continue
        column = _feature_column(feature, rows)
        psi = compute_psi(reference[feature], column)
        sample = reference[feature].get("sample")
        ks = compute_ks(sample, column)
        entry: dict = {"psi": psi, "ks": ks}
        if ks is None:
            entry["ks_reason"] = "no reference sample" if not sample else "no valid values"
        entry["alert"] = drift_alert(psi, ks, args.ks_threshold)
        report["features"][feature] = entry
        report["alert"] = report["alert"] or entry["alert"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if report["alert"] else 0)


if __name__ == "__main__":
    main()
