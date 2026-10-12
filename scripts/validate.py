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
from scripts.evaluate_model import binary_metrics, evaluate_rows, resolve_group_column


def _row_time(row: dict) -> datetime:
    raw = row.get("index_time") or row.get("occurred_at") or ""
    if not raw:
        raise ValueError("every row needs index_time or occurred_at for temporal splitting")
    parsed = datetime.fromisoformat(str(raw))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _contiguous_blocks(groups: list[list[dict]], n_splits: int) -> list[list[dict]]:
    """把有序时间组切成 n_splits + 1 个连续块（组永不拆分；行级与患者级切分共用）。"""
    total = sum(len(group) for group in groups)
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
    return blocks


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
    blocks = _contiguous_blocks(groups, n_splits)
    folds: list[tuple[list[dict], list[dict]]] = []
    for index in range(1, n_splits + 1):
        train = [row for block in blocks[:index] for row in block]
        folds.append((train, blocks[index]))
    return folds


def patient_rolling_time_split(
    rows: list[dict], n_splits: int, group_column: str = "patient_id"
) -> list[tuple[list[dict], list[dict]]]:
    """患者级 expanding window：整患者不跨折、同患者时点不拆块，返回前自证无交集。

    分组键解析 patient_id → encounter_id（默认列缺失时），皆缺 ValueError——
    不静默降级为行级切分（spec §4.2.1）。同一患者时点的多个患者不拆块，沿用
    "时间戳不跨折"裁定；患者时点 = 该患者最早行时间。
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if len(rows) < n_splits + 1:
        raise ValueError(f"need at least {n_splits + 1} rows for {n_splits} splits")
    column = group_column
    if not any(str(row.get(column, "") or "") for row in rows):
        column = resolve_group_column(rows, group_column)  # 回退或响亮报错
    patients: dict[str, list[dict]] = {}
    for row in rows:
        key = str(row.get(column, "") or "")
        if not key:
            raise ValueError(f"every row needs a {column} value for patient-level splitting")
        patients.setdefault(key, []).append(row)
    patient_time = {key: min(_row_time(row) for row in group) for key, group in patients.items()}
    timed: list[tuple[datetime, list[dict]]] = []
    for key in sorted(patients, key=lambda item: (patient_time[item], item)):
        moment = patient_time[key]
        if timed and timed[-1][0] == moment:
            timed[-1][1].extend(patients[key])
        else:
            timed.append((moment, list(patients[key])))
    if len(timed) < n_splits + 1:
        raise ValueError(
            f"need at least {n_splits + 1} distinct patient times so every test fold "
            "starts strictly after its training window"
        )
    blocks = _contiguous_blocks([group for _, group in timed], n_splits)
    folds: list[tuple[list[dict], list[dict]]] = []
    for index in range(1, n_splits + 1):
        train = [row for block in blocks[:index] for row in block]
        test = blocks[index]
        train_keys = {str(row[column]) for row in train}
        test_keys = {str(row[column]) for row in test}
        overlap = train_keys & test_keys
        if overlap:
            # 不变量做成代码而非约定：跨折即断言失败（spec §4.2.4）
            raise AssertionError(f"patient leakage across folds: {sorted(overlap)[:5]}")
        folds.append((train, test))
    return folds


def bootstrap_budget(n_rows: int) -> tuple[int, int | None]:
    """规模裁定（R-RS-10）：>50000 行 → (500 次, 每次 ≤20000 行)；否则 (1000, None) 全量。"""
    if n_rows > 50_000:
        return 500, 20_000
    return 1000, None


def bootstrap_metric(
    labels: list[int],
    scores: list[float],
    metric: Callable[[list[int], list[float]], float],
    n: int = 1000,
    *,
    seed: int | None = None,
    max_rows: int | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap: (point, lo, hi) at 2.5%/97.5% percentiles.

    max_rows：单次 replicate 有放回子采样上限（>50000 行时 ≤20000，方法学仍是
    percentile bootstrap；报告须注明 replicate_subsample，R-RS-10）。
    """
    if not labels or len(labels) != len(scores):
        raise ValueError("labels and scores must be non-empty and the same length")
    rng = Random(seed)
    point = metric(labels, scores)
    size = len(labels)
    sample_size = size if max_rows is None else min(size, max_rows)
    distribution = []
    for _ in range(n):
        indices = [rng.randrange(size) for _ in range(sample_size)]
        distribution.append(metric([labels[i] for i in indices], [scores[i] for i in indices]))
    distribution.sort()
    lo = distribution[max(int(n * 0.025), 0)]
    hi = distribution[min(int(n * 0.975), n - 1)]
    return point, lo, hi


def subgroup_sensitivity_diff(
    labels: list[int],
    scores: list[float],
    groups: list[str],
    threshold: float,
    *,
    min_positives: int = 10,
) -> float | None:
    """两组灵敏度之差绝对值；任一组阳性不足或组数<2 → None（跳过，Ruling R-P7-6）。"""
    by_group: dict[str, tuple[int, int]] = {}
    for label, score, group in zip(labels, scores, groups):
        positives, detected = by_group.get(group, (0, 0))
        if label == 1:
            positives += 1
            if score >= threshold:
                detected += 1
        by_group[group] = (positives, detected)
    sensitivities = []
    for positives, detected in by_group.values():
        if positives < min_positives:
            return None
        sensitivities.append(detected / positives)
    if len(sensitivities) < 2:
        return None
    return abs(sensitivities[0] - sensitivities[1])


def per_facility_auprc(
    rows: list[dict],
    labels: list[int],
    scores: list[float],
    baseline_auprc: float,
    *,
    facility_column: str = "facility_id",
    min_rows: int = 20,
    min_positives: int = 5,
) -> dict:
    """按院区评估（Ruling R-P7-7）：合格院区 AUPRC ≥ baseline 才算过。"""
    buckets: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        facility = str(row.get(facility_column, "") or "")
        if not facility:
            return {"checked": False, "reason": f"missing {facility_column}", "failures": []}
        buckets.setdefault(facility, []).append(index)
    if len(buckets) < 2:
        return {"checked": False, "reason": "fewer than 2 facilities", "failures": []}
    failures = []
    worst = None
    for facility, indices in sorted(buckets.items()):
        if len(indices) < min_rows:
            continue
        facility_labels = [labels[i] for i in indices]
        if sum(facility_labels) < min_positives:
            continue
        auprc = binary_metrics(facility_labels, [scores[i] for i in indices], 0.5)["auprc"] or 0.0
        worst = auprc if worst is None else min(worst, auprc)
        if auprc < baseline_auprc:
            failures.append({"facility": facility, "auprc": auprc, "baseline": baseline_auprc})
    return {"checked": True, "worst": worst, "failures": failures}


def decision_curve_mean_gain(
    labels: list[int],
    scores: list[float],
    grid: list[float] | None = None,
) -> float:
    """mean(NB_model − max(NB_treat_all, 0))（Ruling R-P7-8）。"""
    if not labels or not sum(labels):
        raise ValueError("decision curve needs labels and at least one positive")
    thresholds = grid or [round(0.10 + 0.05 * i, 2) for i in range(17)]
    prevalence = sum(labels) / len(labels)
    gains = []
    for threshold in thresholds:
        tp = sum(1 for score, label in zip(scores, labels) if score >= threshold and label == 1)
        fp = sum(1 for score, label in zip(scores, labels) if score >= threshold and label == 0)
        nb_model = tp / len(labels) - (fp / len(labels)) * (threshold / (1 - threshold))
        nb_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        gains.append(nb_model - max(nb_all, 0.0))
    return sum(gains) / len(gains)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling temporal validation with bootstrap CI")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=None,
        help="replicate 次数；缺省按规模自适应（>50000 行 → 500）",
    )
    parser.add_argument(
        "--bootstrap-max-rows",
        type=int,
        default=None,
        help="replicate 子采样上限；缺省按规模自适应（>50000 行 → 20000）",
    )
    parser.add_argument(
        "--target-id",
        choices=["deterioration_24h", "readmission_30d"],
        default="deterioration_24h",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument(
        "--split",
        choices=["patient", "encounter"],
        default="patient",
        help="患者级（整患者不跨折）或旧行级切分（golden/回归对照用 encounter）",
    )
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
    if args.split == "patient":
        try:
            folds = patient_rolling_time_split(rows, args.n_splits)
        except ValueError as exc:
            raise SystemExit(f"validate: {exc}") from exc
    else:
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

    n_bootstrap = args.n_bootstrap
    max_rows = args.bootstrap_max_rows
    if n_bootstrap is None or max_rows is None:
        auto_n, auto_max = bootstrap_budget(len(oof_labels))
        n_bootstrap = auto_n if n_bootstrap is None else n_bootstrap
        max_rows = auto_max if max_rows is None else max_rows
    point, lo, hi = bootstrap_metric(
        oof_labels, oof_scores, auprc, n=n_bootstrap, seed=20260928, max_rows=max_rows
    )
    report = {
        "target_id": args.target_id,
        "model_version": engine.version,
        "split_mode": args.split,
        "folds": fold_reports,
        "out_of_fold": binary_metrics(oof_labels, oof_scores, threshold),
        "bootstrap_ci": {
            "metric": "auprc",
            "point": point,
            "lo": lo,
            "hi": hi,
            "n": n_bootstrap,
            "replicate_subsample": max_rows,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
