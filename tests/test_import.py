import json

import pytest


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_future_event_rejected(tmp_path, monkeypatch):
    from scripts.import_clinical_data import main as import_main

    patients = _write(
        tmp_path,
        "patients.json",
        [{"patient_id": "pat-1", "encounter_id": "enc-1", "age": 60}],
    )
    events = _write(
        tmp_path,
        "events.json",
        [
            {
                "event_id": "evt-future",
                "event_type": "vital_signs",
                "patient_id": "pat-1",
                "encounter_id": "enc-1",
                "occurred_at": "2099-01-01T00:00:00Z",
                "payload": {"heart_rate": 80},
            }
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        ["import_clinical_data.py", "--patients", str(patients), "--events", str(events)],
    )
    with pytest.raises(ValueError, match="future"):
        import_main()


def test_partial_summary_on_failure(tmp_path, monkeypatch, capsys):
    from app.store import Store
    from app.validation import ImportPartialFailure
    from scripts.import_clinical_data import main as import_main

    patients = _write(
        tmp_path,
        "patients.json",
        [{"patient_id": "pat-1", "encounter_id": "enc-1", "age": 60}],
    )
    events = _write(
        tmp_path,
        "events.json",
        [
            {
                "event_id": "evt-1",
                "event_type": "vital_signs",
                "patient_id": "pat-1",
                "encounter_id": "enc-1",
                "payload": {"heart_rate": 80},
            }
        ],
    )
    summary = {"patients": 1, "accepted_events": 0, "failed": 1, "last_ok": None}

    def boom(self, import_patients, import_events):
        raise ImportPartialFailure(summary, "boom")

    monkeypatch.setattr(Store, "import_records", boom)
    # 隔离 store 路径，避免打开仓库真实 data/store.db
    from app.config import Settings

    monkeypatch.setattr(
        "scripts.import_clinical_data.get_settings",
        lambda: Settings(store_path=str(tmp_path / "isolated.db")),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "import_clinical_data.py",
            "--patients",
            str(patients),
            "--events",
            str(events),
            "--commit",
        ],
    )
    with pytest.raises(ImportPartialFailure):
        import_main()
    out = capsys.readouterr().out
    assert '"accepted_events": 0' in out
    assert '"failed": 1' in out
