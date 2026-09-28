"""Merge label CSVs with seven-tuple landmark features (T5.2)."""

import argparse
import csv
from datetime import UTC, datetime
from pathlib import Path

from app.domain import ClinicalEvent
from app.features import TUPLE_FEATURE_ORDER, build_landmarks
from app.synthetic import demo_patients_with_outcomes


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build landmark features next to a label CSV")
    parser.add_argument("--labels", required=True, help="CSV from derive_labels.py")
    parser.add_argument("--output", default="data/demo_dataset.csv")
    parser.add_argument("--demo", action="store_true", help="regenerate synthetic events (seed=7)")
    parser.add_argument("--window-hours", type=int, default=24)
    args = parser.parse_args()
    if not args.demo:
        raise SystemExit("P5 仅实现 --demo 事件来源；真实数据请调用 build_landmarks()")
    with Path(args.labels).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        label_rows = list(reader)
        label_fields = list(reader.fieldnames or [])
    if not label_rows:
        raise SystemExit("label CSV is empty")
    _, events, _ = demo_patients_with_outcomes()
    patient_events: dict[str, list[ClinicalEvent]] = {}
    for event in events:
        patient_events.setdefault(event.encounter_id, []).append(event)
    index_times = [(row["encounter_id"], _parse_iso(row["index_time"])) for row in label_rows]
    landmarks = {
        (row["encounter_id"], row["index_time"]): row
        for row in build_landmarks(patient_events, index_times, args.window_hours)
    }
    output_fields = label_fields + TUPLE_FEATURE_ORDER
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for row in label_rows:
            landmark = landmarks[(row["encounter_id"], row["index_time"])]
            merged = {field: row.get(field, "") for field in label_fields}
            for field in TUPLE_FEATURE_ORDER:
                value = landmark.get(field)
                merged[field] = "" if value is None else value
            writer.writerow(merged)
    print(f"wrote {output} ({len(label_rows)} rows, {len(TUPLE_FEATURE_ORDER)} feature columns)")


if __name__ == "__main__":
    main()
