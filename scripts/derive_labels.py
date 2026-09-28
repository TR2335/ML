"""Derive landmark labels from outcome events (T5.1).

Demo-first CLI: real-data entry stays the importable derive_labels() function
until the P9 self-service data path lands.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.domain import ClinicalEvent
from app.synthetic import FEATURES, demo_patients_with_outcomes

OUTCOME_TYPES = frozenset({"rrt", "unplanned_icu_transfer", "cardiac_arrest", "death"})
DEMO_INDEX_HOURS = range(1, 97)


@dataclass(frozen=True)
class LabeledLandmark:
    encounter_id: str
    index_time: datetime
    label: int


def _as_datetime(value: datetime | str) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def derive_labels(
    events: list[ClinicalEvent],
    outcomes: list[dict],
    index_times: list[tuple[str, datetime]],
    horizon_hours: int = 24,
) -> list[LabeledLandmark]:
    """Label each (encounter, index) landmark: 1 when a whitelisted outcome
    falls in [index, index + horizon), else 0. Encounters must appear in events.

    index_times is a list of (encounter_id, index_time) pairs so one encounter
    can carry many landmarks (Plan Ruling R13)."""
    encounters = {event.encounter_id for event in events}
    horizon = timedelta(hours=horizon_hours)
    by_encounter: dict[str, list[datetime]] = {}
    for outcome in outcomes:
        if str(outcome.get("outcome_type", "")) not in OUTCOME_TYPES:
            continue
        encounter_id = str(outcome["encounter_id"])
        by_encounter.setdefault(encounter_id, []).append(_as_datetime(outcome["occurred_at"]))
    labeled: list[LabeledLandmark] = []
    for encounter_id, raw_index in sorted(index_times):
        if encounter_id not in encounters:
            continue
        index_time = _as_datetime(raw_index)
        end = index_time + horizon
        label = int(
            any(index_time <= occurred < end for occurred in by_encounter.get(encounter_id, []))
        )
        labeled.append(
            LabeledLandmark(encounter_id=encounter_id, index_time=index_time, label=label)
        )
    return labeled


def demo_index_times(events: list[ClinicalEvent]) -> list[tuple[str, datetime]]:
    """Hourly index grid 1..96h after each encounter's first event (Plan Ruling R3)."""
    first_seen: dict[str, datetime] = {}
    for event in sorted(events, key=lambda item: item.occurred_at):
        first_seen.setdefault(event.encounter_id, event.occurred_at)
    return [
        (encounter_id, first + timedelta(hours=hour))
        for encounter_id, first in first_seen.items()
        for hour in DEMO_INDEX_HOURS
    ]


def demo_labeled_landmarks() -> list[LabeledLandmark]:
    _, events, outcomes = demo_patients_with_outcomes()
    return derive_labels(events, outcomes, demo_index_times(events))


def _latest_payload_before(
    events: list[ClinicalEvent], encounter_id: str, index_time: datetime
) -> dict[str, object]:
    candidates = [
        event
        for event in events
        if event.encounter_id == encounter_id and event.occurred_at < index_time
    ]
    if not candidates:
        return {}
    latest = max(candidates, key=lambda event: event.occurred_at)
    return {
        feature: latest.payload[feature]
        for feature in FEATURES
        if latest.payload.get(feature) is not None
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive landmark labels (demo mode in P5)")
    parser.add_argument("--demo", action="store_true", help="use synthetic patients/outcomes")
    parser.add_argument("--output", default="data/demo_labels.csv")
    parser.add_argument("--horizon-hours", type=int, default=24)
    args = parser.parse_args()
    if not args.demo:
        raise SystemExit(
            "P5 仅实现 --demo；真实数据请调用 derive_labels()（见 docs/data-contract.md）"
        )
    _, events, outcomes = demo_patients_with_outcomes()
    index_times = demo_index_times(events)
    labeled = derive_labels(events, outcomes, index_times, horizon_hours=args.horizon_hours)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["encounter_id", "index_time", "label", *FEATURES])
        for item in labeled:
            payload = _latest_payload_before(events, item.encounter_id, item.index_time)
            writer.writerow(
                [
                    item.encounter_id,
                    item.index_time.isoformat(),
                    item.label,
                    *[payload.get(feature, "") for feature in FEATURES],
                ]
            )
    positives = sum(item.label for item in labeled)
    print(
        json.dumps(
            {
                "rows": len(labeled),
                "positives": positives,
                "positive_rate": round(positives / len(labeled), 4),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
