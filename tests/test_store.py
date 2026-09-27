from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

from app.domain import ClinicalEvent, RiskScore
from app.store import Store


def test_store_round_trip(tmp_path):
    path = tmp_path / "store.db"
    store = Store(str(path))
    patient = {
        "patient_id": "pat-1",
        "encounter_id": "enc-1",
        "display_name": "测试患者",
        "department": "综合内科",
    }
    event = ClinicalEvent(
        event_type="vital_signs",
        patient_id="pat-1",
        encounter_id="enc-1",
        payload={"heart_rate": 110},
    )
    store.seed([patient], [event])
    score = RiskScore(
        patient_id="pat-1", encounter_id="enc-1", model_version="test", score=0.7, band="high"
    )
    store.upsert_score(score)
    reloaded = Store(str(path))
    assert reloaded.get_patient("enc-1", "group-demo", "hospital-01")["risk"]["band"] == "high"
    assert reloaded.events_for("enc-1", "group-demo", "hospital-01")[0].payload["heart_rate"] == 110


def test_rescoring_preserves_human_review(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    event = ClinicalEvent(
        event_type="vital_signs",
        patient_id="pat-1",
        encounter_id="enc-1",
        payload={"heart_rate": 110},
    )
    store.seed(
        [{"patient_id": "pat-1", "encounter_id": "enc-1", "display_name": "测试患者"}], [event]
    )
    score = RiskScore(
        patient_id="pat-1", encounter_id="enc-1", model_version="test", score=0.7, band="high"
    )
    store.upsert_score(score)
    score.status = "resolved"
    score.human_review = "已处置"
    store.upsert_score(score)
    rescored = RiskScore(
        patient_id="pat-1",
        encounter_id="enc-1",
        model_version="test-v2",
        score=0.8,
        band="critical",
    )
    store.upsert_score(rescored)
    result = store.get_score("enc-1", "group-demo", "hospital-01")
    assert result.status == "resolved"
    assert result.human_review == "已处置"
    assert result.model_version == "test-v2"


def test_store_serializes_concurrent_event_writes(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    store.seed([{"patient_id": "pat-1", "encounter_id": "enc-1", "display_name": "测试患者"}], [])

    def write(index: int) -> bool:
        return store.add_event(
            ClinicalEvent(
                event_id=f"evt-{index}",
                event_type="vital_signs",
                patient_id="pat-1",
                encounter_id="enc-1",
                payload={"heart_rate": 80 + index},
            )
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(write, range(20)))

    assert all(results)
    assert len(store.events_for("enc-1", "group-demo", "hospital-01")) == 20


def test_feature_snapshot_excludes_future_events(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    store.seed(
        [{"patient_id": "pat-1", "encounter_id": "enc-1", "display_name": "测试患者"}],
        [],
    )
    now = datetime.now(UTC)
    store.add_event(
        ClinicalEvent(
            event_id="evt-past",
            event_type="vital_signs",
            patient_id="pat-1",
            encounter_id="enc-1",
            occurred_at=now,
            payload={"heart_rate": 80},
        )
    )
    store.add_event(
        ClinicalEvent(
            event_id="evt-future",
            event_type="vital_signs",
            patient_id="pat-1",
            encounter_id="enc-1",
            occurred_at=now + timedelta(hours=1),
            payload={"heart_rate": 140},
        )
    )
    snapshot = store.feature_snapshot("enc-1", "group-demo", "hospital-01", as_of=now)
    assert snapshot["heart_rate"] == 80


def test_event_times_are_normalized_to_utc(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    store.seed(
        [{"patient_id": "pat-1", "encounter_id": "enc-1", "display_name": "测试患者"}],
        [],
    )
    store.add_event(
        ClinicalEvent(
            event_id="evt-utc",
            event_type="vital_signs",
            patient_id="pat-1",
            encounter_id="enc-1",
            occurred_at="2026-09-09T10:00:00+02:00",
            payload={"heart_rate": 100},
        )
    )
    event = store.events_for("enc-1", "group-demo", "hospital-01")[0]
    assert event.occurred_at.isoformat() == "2026-09-09T08:00:00+00:00"


def test_tenant_scoped_encounters_do_not_overwrite_each_other(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    store.seed(
        [
            {
                "patient_id": "pat-a",
                "encounter_id": "shared-encounter-id",
                "tenant_id": "tenant-a",
                "facility_id": "facility-a",
                "display_name": "甲患者",
            },
            {
                "patient_id": "pat-b",
                "encounter_id": "shared-encounter-id",
                "tenant_id": "tenant-b",
                "facility_id": "facility-b",
                "display_name": "乙患者",
            },
        ],
        [],
    )
    store.upsert_score(
        RiskScore(
            patient_id="pat-a",
            encounter_id="shared-encounter-id",
            tenant_id="tenant-a",
            facility_id="facility-a",
            model_version="test",
            score=0.2,
            band="medium",
        )
    )
    store.upsert_score(
        RiskScore(
            patient_id="pat-b",
            encounter_id="shared-encounter-id",
            tenant_id="tenant-b",
            facility_id="facility-b",
            model_version="test",
            score=0.8,
            band="critical",
        )
    )
    assert (
        store.get_patient("shared-encounter-id", "tenant-a", "facility-a")["patient_id"] == "pat-a"
    )
    assert (
        store.get_patient("shared-encounter-id", "tenant-b", "facility-b")["risk"]["band"]
        == "critical"
    )


def test_stale_score_does_not_overwrite(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    store.seed([{"patient_id": "pat-1", "encounter_id": "enc-1", "display_name": "测试患者"}], [])
    base = {"patient_id": "pat-1", "encounter_id": "enc-1", "model_version": "test"}
    newer = RiskScore(
        **base,
        score=0.8,
        band="critical",
        as_of=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
    older = RiskScore(
        **base,
        score=0.2,
        band="low",
        as_of=datetime(2026, 1, 1, 6, 0, tzinfo=UTC),
    )
    assert store.upsert_score(newer) == "applied"
    assert store.upsert_score(older) == "skipped_stale"
    result = store.get_score("enc-1", "group-demo", "hospital-01")
    assert result.score == 0.8
    assert result.band == "critical"


def test_target_scoped_scores_are_independent(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    store.seed([{"patient_id": "pat-1", "encounter_id": "enc-1", "display_name": "测试患者"}], [])
    store.upsert_score(
        RiskScore(
            patient_id="pat-1",
            encounter_id="enc-1",
            model_version="deterioration-v1",
            target_id="deterioration_24h",
            score=0.7,
            band="high",
        )
    )
    store.upsert_score(
        RiskScore(
            patient_id="pat-1",
            encounter_id="enc-1",
            model_version="readmission-v1",
            target_id="readmission_30d",
            score=0.2,
            band="medium",
        )
    )
    scores = store.get_scores("enc-1", "group-demo", "hospital-01")
    assert scores["deterioration_24h"].model_version == "deterioration-v1"
    assert scores["readmission_30d"].model_version == "readmission-v1"
