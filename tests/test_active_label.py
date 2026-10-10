import json


def _engine(tmp_path):
    from app.config import Settings
    from app.risk import RiskEngine
    from app.synthetic import FEATURES

    path = tmp_path / "model.json"
    weights = {feature: 0.0 for feature in FEATURES}
    weights["heart_rate"] = 0.05
    path.write_text(
        json.dumps(
            {
                "model_version": "al-v1",
                "feature_names": FEATURES,
                "weights": weights,
                "bias": -5.0,
                "thresholds": {"medium": 0.2, "high": 0.45, "critical": 0.75},
            }
        ),
        encoding="utf-8",
    )
    return RiskEngine(Settings(model_path=str(path)))


def _rows() -> list[dict]:
    # z = 0.05*hr - 5；hr=100 → z=0 → p=0.5（最不确定）；偏离越远越确定
    return [
        {"encounter_id": f"enc-{hr}", "index_time": "2026-01-01T02:00:00+00:00", "heart_rate": hr}
        for hr in (80.0, 100.0, 120.0, 98.0, 102.0)
    ]


def test_select_for_labeling_orders_by_uncertainty_deterministically(tmp_path):
    from scripts.active_label import select_for_labeling

    engine = _engine(tmp_path)
    first = select_for_labeling(_rows(), engine, top_n=5)
    second = select_for_labeling(list(reversed(_rows())), engine, top_n=5)
    key = lambda row: (row["encounter_id"], row["index_time"])
    assert [key(row) for row in first] == [key(row) for row in second]  # 输入顺序无关
    assert first[0]["heart_rate"] == 100.0  # |p-0.5| 最小者居首
    assert first[0]["uncertainty"] <= first[-1]["uncertainty"]


def test_select_for_labeling_respects_top_n(tmp_path):
    from scripts.active_label import select_for_labeling

    engine = _engine(tmp_path)
    picked = select_for_labeling(_rows(), engine, top_n=2)
    assert len(picked) == 2
    assert all("uncertainty" in row and "score" in row for row in picked)
