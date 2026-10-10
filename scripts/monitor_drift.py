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

PSI_THRESHOLD = 0.2  # WBS T7.4


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
    parser = argparse.ArgumentParser(description="PSI drift check against artifact reference")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--input", required=True, help="CSV batch to compare")
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
        column = [row.get(feature, "") for row in rows]
        psi = compute_psi(reference[feature], column)
        alert = psi > PSI_THRESHOLD
        report["features"][feature] = {"psi": psi, "alert": alert}
        report["alert"] = report["alert"] or alert
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if report["alert"] else 0)


if __name__ == "__main__":
    main()
