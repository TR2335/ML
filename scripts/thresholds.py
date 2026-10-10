"""Derive band cutpoints from a validation set (Plan Ruling R6)."""

import argparse
import csv
import json
import sys
from pathlib import Path

# 直接以 `python scripts/x.py` 运行时根目录不在 sys.path（Ruling R14）。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.risk import RiskEngine
from scripts.evaluate_model import evaluate_rows


def _max_threshold_at_tpr(scores: list[float], labels: list[int], tpr_target: float) -> float:
    positives = [s for s, y in zip(scores, labels) if y == 1]
    if not positives:
        raise ValueError("validation set has no positives; cannot derive thresholds")
    best = None
    for candidate in sorted(set(scores)):
        tpr = sum(1 for s in positives if s >= candidate) / len(positives)
        if tpr >= tpr_target - 1e-12:
            best = candidate
    if best is None:
        raise ValueError("threshold: no cutpoint satisfies the sensitivity target")
    return best


def derive_thresholds(
    scores: list[float], labels: list[int], target_sensitivity: float = 0.90
) -> dict[str, float]:
    """medium/high/critical = TPR ladder (S, 0.75S, 0.5S), strictly ascending (Ruling R6)."""
    ladder = (
        ("medium", target_sensitivity),
        ("high", target_sensitivity * 0.75),
        ("critical", target_sensitivity * 0.5),
    )
    result: dict[str, float] = {}
    previous = float("-inf")
    distinct = sorted(set(scores))
    for name, tpr in ladder:
        candidate = _max_threshold_at_tpr(scores, labels, tpr)
        while candidate <= previous:
            higher = [s for s in distinct if s > candidate]
            if not higher:
                raise ValueError(
                    "threshold: cannot form strictly ascending cutpoints from this score distribution"
                )
            candidate = higher[0]
        result[name] = candidate
        previous = candidate
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite artifact thresholds from a holdout CSV")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--target-sensitivity", type=float, default=0.90)
    args = parser.parse_args()
    engine = RiskEngine(Settings(model_path=args.artifact, app_env="dev"))
    with Path(args.input).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    labels, scores = evaluate_rows(rows, engine, args.label_column)
    thresholds = derive_thresholds(scores, labels, args.target_sensitivity)
    artifact_path = Path(args.artifact)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["thresholds"] = thresholds
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps({"artifact": str(artifact_path), "thresholds": thresholds}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
