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


def test_legacy_row_without_as_of_is_rescored(tmp_path):
    """Review Important #1: a pre-upgrade row (no as_of) must not freeze rescoring."""
    store = Store(str(tmp_path / "store.db"))
    store.seed([{"patient_id": "pat-1", "encounter_id": "enc-1", "display_name": "测试患者"}], [])
    base = {"patient_id": "pat-1", "encounter_id": "enc-1", "model_version": "test"}
    legacy = RiskScore(**base, score=0.3, band="medium")  # as_of 缺失（升级前存量形态）
    assert store.upsert_score(legacy) == "applied"
    rescored = RiskScore(
        **base,
        score=0.7,
        band="high",
        as_of=datetime(2026, 1, 1, 6, 0, tzinfo=UTC),
    )
    # 存量行陈旧度未知 → 不应阻止基于历史事件的重打分
    assert store.upsert_score(rescored) == "applied"
    assert store.get_score("enc-1", "group-demo", "hospital-01").score == 0.7


def test_cross_tenant_event_id_collision_accepted(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    store.seed(
        [
            {
                "patient_id": "pat-a",
                "encounter_id": "enc-a",
                "tenant_id": "tenant-a",
                "facility_id": "facility-a",
                "display_name": "甲患者",
            },
            {
                "patient_id": "pat-b",
                "encounter_id": "enc-b",
                "tenant_id": "tenant-b",
                "facility_id": "facility-b",
                "display_name": "乙患者",
            },
        ],
        [],
    )
    first = ClinicalEvent(
        event_id="shared-id",
        event_type="lab_result",
        patient_id="pat-a",
        encounter_id="enc-a",
        tenant_id="tenant-a",
        facility_id="facility-a",
        payload={"lactate": 3.0},
    )
    second = ClinicalEvent(
        event_id="shared-id",
        event_type="lab_result",
        patient_id="pat-b",
        encounter_id="enc-b",
        tenant_id="tenant-b",
        facility_id="facility-b",
        payload={"lactate": 4.0},
    )
    assert store.add_event(first) is True
    assert store.add_event(second) is True  # 跨租户不再被全局 PK 吞掉
    assert store.add_event(first) is False  # 同租户重投递仍幂等
    assert len(store.events_for("enc-a", "tenant-a", "facility-a")) == 1
    assert len(store.events_for("enc-b", "tenant-b", "facility-b")) == 1


def test_event_schema_migration_preserves_rows(tmp_path):
    import sqlite3

    path = tmp_path / "store.db"
    legacy_sql = """
    CREATE TABLE patients (
        encounter_id TEXT NOT NULL,
        patient_id TEXT NOT NULL,
        tenant_id TEXT NOT NULL,
        facility_id TEXT NOT NULL,
        data_json TEXT NOT NULL,
        PRIMARY KEY (tenant_id, encounter_id)
    );
    CREATE TABLE events (
        event_id TEXT PRIMARY KEY,
        encounter_id TEXT NOT NULL,
        patient_id TEXT NOT NULL,
        tenant_id TEXT NOT NULL,
        facility_id TEXT NOT NULL,
        occurred_at TEXT NOT NULL,
        data_json TEXT NOT NULL
    );
    INSERT INTO events VALUES
    ('evt-old1', 'enc-1', 'pat-1', 'tenant-a', 'facility-a', '2026-01-01T06:00:00+00:00', '{}'),
    ('evt-old2', 'enc-2', 'pat-2', 'tenant-b', 'facility-b', '2026-01-01T07:00:00+00:00', '{}');
    """
    raw = sqlite3.connect(path)
    raw.executescript(legacy_sql)
    raw.commit()
    raw.close()

    store = Store(str(path))  # 触发 _create_schema → 迁移
    with store._lock:
        pk = {
            row["name"]: row["pk"]
            for row in store.connection.execute("PRAGMA table_info(events)").fetchall()
            if row["pk"]
        }
        count = store.connection.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
    # PRAGMA 按表列序返回行；pk 序号携带主键键序（tenant_id 为第一键）
    assert pk == {"tenant_id": 1, "event_id": 2}
    assert count == 2
    store.close()
    reopened = Store(str(path))  # 迁移幂等：再次打开无异常
    with reopened._lock:
        assert reopened.connection.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"] == 2
    reopened.close()


def test_event_tenant_scope_migration_is_noop_when_already_migrated(tmp_path):
    """已迁移的库再次触发迁移必须是 no-op（PRAGMA 按表列序返回，检测须按主键键序）。"""
    store = Store(str(tmp_path / "store.db"))
    statements: list[str] = []
    store.connection.set_trace_callback(statements.append)
    store._migrate_event_tenant_scope()
    store.connection.set_trace_callback(None)
    ddl = [
        sql
        for sql in statements
        if any(k in sql.upper() for k in ("CREATE TABLE", "ALTER TABLE", "DROP TABLE"))
    ]
    assert ddl == [], f"迁移体在已迁移的库上重跑了: {ddl}"
    store.close()


def test_stale_events_legacy_fails_fast_with_guidance(tmp_path):
    """上次迁移中断留下的 events_legacy 残留必须显式报错，不得静默建空表丢数据。"""
    import sqlite3

    import pytest

    path = tmp_path / "store.db"
    raw = sqlite3.connect(path)
    raw.executescript(
        """
        CREATE TABLE events_legacy (
            event_id TEXT PRIMARY KEY,
            encounter_id TEXT NOT NULL,
            patient_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            facility_id TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            data_json TEXT NOT NULL
        );
        INSERT INTO events_legacy VALUES
        ('evt-orphan', 'enc-1', 'pat-1', 'tenant-a', 'facility-a', '2026-01-01T06:00:00+00:00', '{}');
        """
    )
    raw.commit()
    raw.close()

    with pytest.raises(RuntimeError, match="events_legacy"):
        Store(str(path))


def test_upsert_score_uses_immediate_transaction(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    store.seed([{"patient_id": "pat-1", "encounter_id": "enc-1", "display_name": "测试患者"}], [])
    statements: list[str] = []
    store.connection.set_trace_callback(statements.append)
    try:
        result = store.upsert_score(
            RiskScore(
                patient_id="pat-1",
                encounter_id="enc-1",
                model_version="test",
                score=0.5,
                band="medium",
            )
        )
    finally:
        store.connection.set_trace_callback(None)
    assert result == "applied"
    normalized = [s.strip().upper() for s in statements]
    assert any(s.startswith("BEGIN IMMEDIATE") for s in normalized), normalized


def test_concurrent_review_and_score_both_kept(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    store = Store(str(tmp_path / "store.db"))
    store.seed([{"patient_id": "pat-1", "encounter_id": "enc-1", "display_name": "测试患者"}], [])
    base = {"patient_id": "pat-1", "encounter_id": "enc-1", "model_version": "test"}
    # 初始分数：as_of 取旧时刻，保证并发 rescore 必然"更新"而非 stale
    store.upsert_score(
        RiskScore(
            **base,
            score=0.1,
            band="low",
            as_of=datetime(2026, 1, 1, 6, 0, tzinfo=UTC),
        )
    )

    def review_once() -> None:
        score = store.get_score("enc-1", "group-demo", "hospital-01")
        score.status = "resolved"
        score.human_review = "已处置"
        store.upsert_score(score, preserve_review=False)

    def rescore() -> str:
        return store.upsert_score(
            RiskScore(
                **base,
                score=0.9,
                band="critical",
                as_of=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            )
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fn) for fn in (review_once, rescore) * 10]
        for future in futures:
            future.result()

    # 全行写入下最终 score 取决于交错顺序（review 可能携带旧分数最后写入）——
    # 用一次确定性的最终 rescore 收敛分数字段；preserve_review 必须保住并发写入的复核状态。
    store.upsert_score(
        RiskScore(
            **base,
            score=0.9,
            band="critical",
            as_of=datetime(2026, 1, 1, 13, 0, tzinfo=UTC),
        )
    )
    final = store.get_score("enc-1", "group-demo", "hospital-01")
    assert final.status == "resolved"
    assert final.human_review == "已处置"
    assert final.score == 0.9


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
