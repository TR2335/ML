"""Temporal validation: rolling time split + bootstrap confidence intervals."""

import argparse
import csv
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from random import Random

# 直接以 `python scripts/x.py` 运行时根目录不在 sys.path，补一次以便跨 scripts 导入。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.risk import RiskEngine
from scripts.evaluate_model import binary_metrics, evaluate_rows


def _row_time(row: dict) -> datetime:
    raw = row.get("index_time") or row.get("occurred_at") or ""
    if not raw:
        raise ValueError("every row needs index_time or occurred_at for temporal splitting")
    parsed = datetime.fromisoformat(str(raw))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def rolling_time_split(rows: list[dict], n_splits: int) -> list[tuple[list[dict], list[dict]]]:
    """Expanding-window folds over time-ordered rows.

    Rows are sorted by time and grouped by timestamp; groups are cut into
    n_splits + 1 contiguous blocks (a timestamp never straddles a boundary),
    so fold i trains on blocks 0..i-1 and tests on block i: test blocks are
    disjoint and every train row is strictly earlier than every test row —
    including when several rows share one timestamp (Review Focus #5).
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if len(rows) < n_splits + 1:
        raise ValueError(f"need at least {n_splits + 1} rows for {n_splits} splits")
    ordered = sorted(rows, key=_row_time)
    groups: list[list[dict]] = []
    for row in ordered:
        if groups and _row_time(groups[-1][0]) == _row_time(row):
            groups[-1].append(row)
        else:
            groups.append([row])
    if len(groups) < n_splits + 1:
        raise ValueError(
            f"need at least {n_splits + 1} distinct timestamps so every test fold "
            "starts strictly after its training window"
        )
    total = len(ordered)
    blocks: list[list[dict]] = []
    group_index = 0
    assigned = 0
    remaining = len(groups)
    for block_i in range(1, n_splits + 2):
        blocks_after = n_splits + 1 - block_i
        target = block_i * total / (n_splits + 1)
        block: list[dict] = []
        while remaining > blocks_after:
            if block and assigned + len(block) >= target:
                break
            block.extend(groups[group_index])
            group_index += 1
            remaining -= 1
        assigned += len(block)
        blocks.append(block)
    folds: list[tuple[list[dict], list[dict]]] = []
    for index in range(1, n_splits + 1):
        train = [row for block in blocks[:index] for row in block]
        folds.append((train, blocks[index]))
    return folds


def bootstrap_metric(
    labels: list[int],
    scores: list[float],
    metric: Callable[[list[int], list[float]], float],
    n: int = 1000,
    *,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap: (point, lo, hi) at 2.5%/97.5% percentiles."""
    if not labels or len(labels) != len(scores):
        raise ValueError("labels and scores must be non-empty and the same length")
    rng = Random(seed)
    point = metric(labels, scores)
    size = len(labels)
    distribution = []
    for _ in range(n):
        indices = [rng.randrange(size) for _ in range(size)]
        distribution.append(metric([labels[i] for i in indices], [scores[i] for i in indices]))
    distribution.sort()
    lo = distribution[max(int(n * 0.025), 0)]
    hi = distribution[min(int(n * 0.975), n - 1)]
    return point, lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling temporal validation with bootstrap CI")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--target-id",
        choices=["deterioration_24h", "readmission_30d"],
        default="deterioration_24h",
    )
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()
    model_path = args.model or Path(
        "data/model.json"
        if args.target_id == "deterioration_24h"
        else "data/readmission_model.json"
    )
    engine = RiskEngine(
        Settings(model_path=str(model_path), app_env="dev"), target_id=args.target_id
    )
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    folds = rolling_time_split(rows, args.n_splits)
    fold_reports = []
    oof_labels: list[int] = []
    oof_scores: list[float] = []
    for train, test in folds:
        labels, scores = evaluate_rows(test, engine, args.label_column)
        oof_labels.extend(labels)
        oof_scores.extend(scores)
        fold_reports.append(
            {
                "train_rows": len(train),
                "test_rows": len(test),
                "evaluation": binary_metrics(labels, scores, engine.thresholds["high"]),
            }
        )
    threshold = engine.thresholds["high"]

    def auprc(labels: list[int], scores: list[float]) -> float:
        return binary_metrics(labels, scores, threshold)["auprc"] or 0.0

    point, lo, hi = bootstrap_metric(
        oof_labels, oof_scores, auprc, n=args.n_bootstrap, seed=20260928
    )
    report = {
        "target_id": args.target_id,
        "model_version": engine.version,
        "folds": fold_reports,
        "out_of_fold": binary_metrics(oof_labels, oof_scores, threshold),
        "bootstrap_ci": {
            "metric": "auprc",
            "point": point,
            "lo": lo,
            "hi": hi,
            "n": args.n_bootstrap,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
