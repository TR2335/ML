"""Validate and import de-identified patient and event exports.

The importer accepts a JSON array or newline-delimited JSON (NDJSON). It is
dry-run by default; pass ``--commit`` only after the validation summary has
been reviewed. Production feeds should call the same contract through the
hospital integration layer and preserve source identifiers for lineage.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.domain import ClinicalEvent, PatientRecord
from app.storage import create_store


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"{path} must contain a JSON array")
        return value
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} must be an object")
        records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--commit", action="store_true", help="write validated records")
    args = parser.parse_args()

    patients = [
        PatientRecord.model_validate(item).model_dump(mode="json")
        for item in load_records(args.patients)
    ]
    events = [ClinicalEvent.model_validate(item) for item in load_records(args.events)]
    patient_keys = {
        (item["tenant_id"], item["facility_id"], item["encounter_id"]): item["patient_id"]
        for item in patients
    }
    for event in events:
        key = (event.tenant_id, event.facility_id, event.encounter_id)
        if patient_keys.get(key) != event.patient_id:
            raise ValueError(f"event {event.event_id} does not match a patient record for {key}")

    summary = {
        "patients": len(patients),
        "events": len(events),
        "commit": args.commit,
        "event_types": sorted({event.event_type for event in events}),
    }
    if args.commit:
        store = create_store(get_settings())
        try:
            for patient in patients:
                store.upsert_patient(patient)
            accepted = 0
            duplicates = 0
            for event in events:
                if store.add_event(event):
                    accepted += 1
                else:
                    duplicates += 1
            summary.update({"accepted_events": accepted, "duplicate_events": duplicates})
        finally:
            store.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
