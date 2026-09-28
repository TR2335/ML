from datetime import UTC, datetime, timedelta

from app.domain import ClinicalEvent

INDEX = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _vitals(at: datetime, heart_rate: float) -> ClinicalEvent:
    return ClinicalEvent(
        event_type="vital_signs",
        patient_id="pat-1",
        encounter_id="enc-1",
        occurred_at=at,
        payload={"heart_rate": heart_rate},
    )


def test_landmark_slope_linear_series():
    from app.features import build_landmarks

    # heart_rate 合法区间 (20, 300)，取 100/110/120 构造每小时 +10 的线性序列
    events = [
        _vitals(INDEX - timedelta(hours=3), 100),
        _vitals(INDEX - timedelta(hours=2), 110),
        _vitals(INDEX - timedelta(hours=1), 120),
    ]
    rows = build_landmarks({"enc-1": events}, [("enc-1", INDEX)])
    assert len(rows) == 1
    # 每小时上升 10 → 最小二乘斜率 10.0/小时
    assert rows[0]["heart_rate__slope"] == 10.0
    assert rows[0]["heart_rate__latest"] == 120
    assert rows[0]["heart_rate__missing"] == 0.0
    assert rows[0]["heart_rate__since_last"] == 1.0


def test_landmark_missing_feature_flag():
    from app.features import build_landmarks

    rows = build_landmarks({"enc-1": []}, [("enc-1", INDEX)])
    assert rows[0]["heart_rate__missing"] == 1.0
    assert rows[0]["heart_rate__latest"] is None
    assert rows[0]["heart_rate__slope"] is None
    assert rows[0]["heart_rate__since_last"] is None


def test_landmark_window_excludes_events_at_index():
    from app.features import build_landmarks

    # 恰在 index 时刻的测量属于未来信息，必须被排除（防泄漏）
    rows = build_landmarks({"enc-1": [_vitals(INDEX, 130)]}, [("enc-1", INDEX)])
    assert rows[0]["heart_rate__missing"] == 1.0


def test_landmark_window_includes_lower_bound():
    from app.features import build_landmarks

    # 恰在 index-24h 下界的测量属于窗口内（半开区间下界闭）
    rows = build_landmarks(
        {"enc-1": [_vitals(INDEX - timedelta(hours=24), 77)]}, [("enc-1", INDEX)]
    )
    assert rows[0]["heart_rate__missing"] == 0.0
    assert rows[0]["heart_rate__latest"] == 77


def test_feature_order_includes_news2_score():
    from app.features import FEATURE_ORDER

    assert FEATURE_ORDER[-1] == "news2_score"
    assert len(FEATURE_ORDER) == 71
