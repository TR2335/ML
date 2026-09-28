from datetime import UTC, datetime, timedelta

from app.domain import ClinicalEvent

BASE = datetime(2026, 1, 1, 6, 0, tzinfo=UTC)


def _event(encounter_id: str = "enc-1", occurred_at: datetime = BASE) -> ClinicalEvent:
    return ClinicalEvent(
        event_type="vital_signs",
        patient_id="pat-1",
        encounter_id=encounter_id,
        occurred_at=occurred_at,
        payload={"heart_rate": 80},
    )


def _outcome(occurred_at: datetime, encounter_id: str = "enc-1", kind: str = "rrt") -> dict:
    return {
        "encounter_id": encounter_id,
        "occurred_at": occurred_at.isoformat(),
        "outcome_type": kind,
    }


def test_label_window_boundary_strict():
    from scripts.derive_labels import derive_labels

    index = BASE + timedelta(hours=24)
    near = derive_labels(
        [_event()], [_outcome(index + timedelta(hours=23, minutes=59))], [("enc-1", index)]
    )
    edge = derive_labels([_event()], [_outcome(index + timedelta(hours=24))], [("enc-1", index)])
    on_index = derive_labels([_event()], [_outcome(index)], [("enc-1", index)])
    assert near[0].label == 1  # 严格 < 24h 阳性
    assert edge[0].label == 0  # 恰 24h 整点阴性
    assert on_index[0].label == 1  # 下界含 index 时刻


def test_outcome_outside_window_ignored():
    from scripts.derive_labels import derive_labels

    index = BASE + timedelta(hours=24)
    far = derive_labels([_event()], [_outcome(index + timedelta(hours=48))], [("enc-1", index)])
    unknown = derive_labels(
        [_event()], [_outcome(index + timedelta(hours=1), kind="lab_work")], [("enc-1", index)]
    )
    assert far[0].label == 0
    assert unknown[0].label == 0  # 白名单外的结局类型不计入


def test_no_outcome_is_zero():
    from scripts.derive_labels import derive_labels

    labeled = derive_labels([_event()], [], [("enc-1", BASE)])
    assert [item.label for item in labeled] == [0]


def test_demo_positive_rate_in_range():
    from scripts.derive_labels import demo_labeled_landmarks

    labeled = demo_labeled_landmarks()
    positives = sum(item.label for item in labeled)
    rate = positives / len(labeled)
    assert len(labeled) == 18 * 96
    assert 0.05 <= rate <= 0.15, f"demo 阳性率 {rate:.4f} 超出 5–15%"


def test_demo_csv_output(tmp_path, monkeypatch, capsys):
    import csv
    import json

    from app.synthetic import FEATURES
    from scripts.derive_labels import main

    output = tmp_path / "labels.csv"
    monkeypatch.setattr("sys.argv", ["derive_labels.py", "--demo", "--output", str(output)])
    main()
    summary = json.loads(capsys.readouterr().out)
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == ["encounter_id", "index_time", "label", *FEATURES]
    assert len(rows) == 18 * 96
    assert summary["positives"] == sum(int(row["label"]) for row in rows)
    assert 0.05 <= summary["positive_rate"] <= 0.15
