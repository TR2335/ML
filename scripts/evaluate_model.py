"""Evaluate a target-specific model on a labeled CSV export.

This intentionally uses only the Python standard library so a validation run
can be reproduced inside a restricted hospital environment. The CSV must
contain the model features plus a binary label column; optional timestamp and
group columns support temporal holdout and subgroup reporting.
"""

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

# 直接以 `python scripts/x.py` 运行时根目录不在 sys.path，补一次以便跨 scripts 导入。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.domain import ClinicalEvent
from app.risk import RiskEngine
from app.synthetic import FEATURES


def _expected_calibration_error(
    labels: list[int], scores: list[float], bins: int = 10
) -> float | None:
    if not labels:
        return None
    total = len(labels)
    buckets: list[list[tuple[int, float]]] = [[] for _ in range(bins)]
    for label, score in zip(labels, scores):
        buckets[min(int(score * bins), bins - 1)].append((label, score))
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        confidence = sum(score for _, score in bucket) / len(bucket)
        accuracy = sum(label for label, _ in bucket) / len(bucket)
        ece += len(bucket) / total * abs(accuracy - confidence)
    return ece


def binary_metrics(
    labels: list[int], scores: list[float], threshold: float
) -> dict[str, float | int | None]:
    positives = sum(labels)
    negatives = len(labels) - positives
    pairs = sorted(zip(scores, labels), reverse=True)
    ascending = sorted(zip(scores, labels))
    rank_sum = 0.0
    index = 0
    while index < len(ascending):
        end = index + 1
        while end < len(ascending) and ascending[end][0] == ascending[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(label for _, label in ascending[index:end])
        index = end
    auroc = (
        None
        if not positives or not negatives
        else (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    )
    tp = sum(label == 1 and score >= threshold for label, score in zip(labels, scores))
    fp = sum(label == 0 and score >= threshold for label, score in zip(labels, scores))
    tn = negatives - fp
    precision_recall = []
    for index, (_, label) in enumerate(pairs, start=1):
        true_positive = sum(item_label for _, item_label in pairs[:index])
        precision_recall.append(
            (true_positive / index, true_positive / positives if positives else 0)
        )
    auprc = 0.0
    previous_recall = 0.0
    for precision, recall in precision_recall:
        auprc += precision * max(recall - previous_recall, 0)
        previous_recall = recall
    return {
        "n": len(labels),
        "prevalence": positives / len(labels) if labels else None,
        "auroc": auroc,
        "auprc": auprc if positives else None,
        "brier": sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels)
        if labels
        else None,
        "ece": _expected_calibration_error(labels, scores),
        "sensitivity": tp / positives if positives else None,
        "specificity": tn / negatives if negatives else None,
        "ppv": tp / (tp + fp) if tp + fp else None,
    }


def evaluate_rows(
    rows: Iterable[dict[str, str]], engine: RiskEngine, label_column: str
) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    scores: list[float] = []
    for index, row in enumerate(rows, start=1):
        try:
            label = int(row[label_column])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"row {index}: {label_column} must be 0 or 1") from exc
        if label not in (0, 1):
            raise ValueError(f"row {index}: {label_column} must be 0 or 1")
        payload = {
            feature: float(row[feature]) for feature in FEATURES if row.get(feature, "") != ""
        }
        if row.get("news2_score", "") not in ("", None):
            payload["news2_score"] = float(row["news2_score"])
        event = ClinicalEvent(
            event_type="vital_signs",
            patient_id=row.get("patient_id", f"validation-patient-{index}"),
            encounter_id=row.get("encounter_id", f"validation-encounter-{index}"),
            occurred_at=row.get("occurred_at") or datetime.now(UTC),
            payload=payload,
        )
        labels.append(label)
        scores.append(engine.score(event).score)
    return labels, scores


def _row_time(row: dict[str, str]) -> datetime:
    raw = row.get("index_time") or row.get("occurred_at")
    if not raw:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(raw)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _payload(row: dict[str, str], include_news2: bool) -> dict[str, float]:
    payload = {
        feature: float(row[feature])
        for feature in FEATURES
        if row.get(feature, "") not in ("", None)
    }
    if include_news2 and row.get("news2_score", "") not in ("", None):
        payload["news2_score"] = float(row["news2_score"])
    return payload


def _make_event(
    row: dict[str, str], payload: dict[str, float], occurred_at: datetime, index: int
) -> ClinicalEvent:
    return ClinicalEvent(
        event_type="vital_signs",
        patient_id=row.get("patient_id", f"validation-patient-{index}"),
        encounter_id=row.get("encounter_id", f"validation-encounter-{index}"),
        occurred_at=occurred_at,
        payload=payload,
    )


def comparison_metrics(
    rows: Iterable[dict[str, str]], engine: RiskEngine, label_column: str
) -> dict[str, dict[str, float | None]]:
    """Three-way metrics on the same rows/labels: model (news2 median-imputed
    for v2 artifacts), news2 baseline (normalized /15, threshold 7/15),
    model_with_news2 (news2_score provided in the payload)."""
    from scripts.baselines import news2_score

    model_labels: list[int] = []
    model_scores: list[float] = []
    enriched_labels: list[int] = []
    enriched_scores: list[float] = []
    baseline_labels: list[int] = []
    baseline_scores: list[float] = []
    for index, row in enumerate(rows, start=1):
        try:
            label = int(row[label_column])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"row {index}: {label_column} must be 0 or 1") from exc
        if label not in (0, 1):
            raise ValueError(f"row {index}: {label_column} must be 0 or 1")
        occurred_at = _row_time(row)
        model_labels.append(label)
        model_scores.append(
            engine.score(_make_event(row, _payload(row, False), occurred_at, index)).score
        )
        enriched_labels.append(label)
        enriched_scores.append(
            engine.score(
                _make_event(row, _payload(row, include_news2=True), occurred_at, index)
            ).score
        )
        total, _ = news2_score(_payload(row, False))
        baseline_labels.append(label)
        baseline_scores.append(total / 15)
    return {
        "model": binary_metrics(model_labels, model_scores, engine.thresholds["high"]),
        "news2": binary_metrics(baseline_labels, baseline_scores, 7 / 15),
        "model_with_news2": binary_metrics(
            enriched_labels, enriched_scores, engine.thresholds["high"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--group-column", default=None)
    parser.add_argument(
        "--split-time", default=None, help="ISO timestamp; evaluate only rows at/after this time"
    )
    parser.add_argument(
        "--input-format",
        choices=["default", "labeled_csv"],
        default="default",
        help="labeled_csv: use index_time column and three-way news2 comparison",
    )
    parser.add_argument(
        "--target-id", choices=["deterioration_24h", "readmission_30d"], default="deterioration_24h"
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
    if args.split_time:
        split = datetime.fromisoformat(args.split_time).astimezone(UTC)
        rows = [
            row
            for row in rows
            if row.get("occurred_at")
            and datetime.fromisoformat(row["occurred_at"]).astimezone(UTC) >= split
        ]
    report: dict[str, object] = {
        "target_id": args.target_id,
        "model_version": engine.version,
        "thresholds": engine.thresholds,
    }
    if args.input_format == "labeled_csv":
        if args.group_column:
            raise SystemExit("subgroup analysis requires --input-format default (Plan Ruling R12)")
        comparison = comparison_metrics(rows, engine, args.label_column)
        report["evaluation"] = comparison["model"]
        report["comparison"] = comparison
        has_news2 = "news2_score" in engine.feature_order
        delta = (
            (comparison["model_with_news2"]["auprc"] or 0.0) - (comparison["model"]["auprc"] or 0.0)
            if has_news2
            else None
        )
        report["news2_feature_gain"] = {
            "auprc_model": comparison["model"]["auprc"],
            "auprc_model_with_news2": comparison["model_with_news2"]["auprc"],
            "delta": None if delta is None else round(delta, 6),
            "recommendation": (
                "unsupported" if not has_news2 else "keep" if delta and delta > 0.001 else "drop"
            ),
        }
    else:
        labels, scores = evaluate_rows(rows, engine, args.label_column)
        report["evaluation"] = binary_metrics(labels, scores, engine.thresholds["high"])
    if args.group_column:
        groups = sorted({row.get(args.group_column, "unknown") for row in rows})
        report["subgroups"] = {}
        for group in groups:
            group_rows = [row for row in rows if row.get(args.group_column, "unknown") == group]
            group_labels, group_scores = evaluate_rows(group_rows, engine, args.label_column)
            report["subgroups"][group] = binary_metrics(
                group_labels, group_scores, engine.thresholds["high"]
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
