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


def test_discharge_snapshot_fields_match_readmission_payload_keys():
    """P8 deferred 5：两处 6 字段契约必须集合相等——双常量漂移由本测试拦截。"""
    from app.domain import DISCHARGE_SNAPSHOT_FIELDS
    from app.features import READMISSION_PAYLOAD_KEYS

    assert len(DISCHARGE_SNAPSHOT_FIELDS) == 6
    assert set(DISCHARGE_SNAPSHOT_FIELDS) == set(READMISSION_PAYLOAD_KEYS)


def test_encode_readmission_payload_maps_fields_and_handles_missing():
    import math

    from app.features import READMISSION_FEATURES, encode_readmission_payload

    encoded = encode_readmission_payload(
        {
            "los_days": 5.0,
            "prior_utilization_12m": 2,
            "comorbidity_index": 3.0,
            "ed_visits_6m": 1,
            "diagnosis_codes": ["I50.9", "N17.9"],
            "discharge_destination": "snf",
        }
    )
    assert len(encoded) == len(READMISSION_FEATURES) == 6
    assert encoded == [5.0, 2.0, 3.0, 1.0, 2.0, 2.0]
    partial = encode_readmission_payload({"discharge_destination": "home"})
    assert all(math.isnan(value) for value in partial[:5])  # NaN≠NaN，不能直接比列表
    assert partial[5] == 0.0
    # JSON 字符串形式的诊断列表（CSV 路径）与 list 等价
    from_list = encode_readmission_payload({"diagnosis_codes": ["A", "B", "C"]})
    from_json = encode_readmission_payload({"diagnosis_codes": '["A", "B", "C"]'})
    assert from_list[4] == from_json[4] == 3.0
    # destination 枚举序数：home=0, home_health=1, snf=2, other=3
    assert encode_readmission_payload({"discharge_destination": "home_health"})[5] == 1.0
    assert encode_readmission_payload({"discharge_destination": "other"})[5] == 3.0


def test_payload_from_row_skips_meta_and_parses_json():
    from scripts.evaluate_model import payload_from_row

    payload = payload_from_row(
        {
            "encounter_id": "enc-1",
            "index_time": "2026-01-01T00:00:00+00:00",
            "label": "1",
            "offline_score": "0.5",
            "serving_score": "0.5",
            "heart_rate": "80",
            "discharge_destination": "home",
            "diagnosis_codes": '["I50.9"]',
            "note_column": "",
            "empty": None,
        }
    )
    assert set(payload) == {"heart_rate", "discharge_destination", "diagnosis_codes"}
    assert payload["heart_rate"] == 80.0
    assert payload["diagnosis_codes"] == ["I50.9"]
    assert isinstance(payload["discharge_destination"], str)
