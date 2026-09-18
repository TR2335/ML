"""Evaluate a target-specific model on a labeled CSV export.

This intentionally uses only the Python standard library so a validation run
can be reproduced inside a restricted hospital environment. The CSV must
contain the model features plus a binary label column; optional timestamp and
group columns support temporal holdout and subgroup reporting.
"""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from app.config import Settings
from app.domain import ClinicalEvent
from app.risk import RiskEngine
from app.synthetic import FEATURES


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--group-column", default=None)
    parser.add_argument(
        "--split-time", default=None, help="ISO timestamp; evaluate only rows at/after this time"
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
    labels, scores = evaluate_rows(rows, engine, args.label_column)
    report: dict[str, object] = {
        "target_id": args.target_id,
        "model_version": engine.version,
        "thresholds": engine.thresholds,
        "evaluation": binary_metrics(labels, scores, engine.thresholds["high"]),
    }
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
