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


def test_engine_loader_uses_given_artifact_for_readmission(tmp_path):
    """Review I1：--model 必须对 readmission 目标生效（此前被默认路径覆盖）。"""
    from app.synthetic import FEATURES
    from scripts.active_label import _engine_for

    path = tmp_path / "rm.json"
    weights = {feature: 0.0 for feature in FEATURES}
    weights["creatinine"] = 0.5
    path.write_text(
        json.dumps(
            {
                "model_version": "rm-loader-v1",
                "target_id": "readmission_30d",
                "feature_names": FEATURES,
                "weights": weights,
                "bias": -2.0,
                "thresholds": {"medium": 0.15, "high": 0.3, "critical": 0.55},
            }
        ),
        encoding="utf-8",
    )
    engine = _engine_for(str(path), "readmission_30d")
    assert engine.version == "rm-loader-v1"
    assert engine.target_id == "readmission_30d"


def test_active_label_cli_dedupes_score_columns(tmp_path, monkeypatch):
    """P6 deferred 6：输入已含 score/uncertainty（对上轮输出再跑）→ 输出表头不重复。"""
    import sys

    from app.synthetic import FEATURES
    from scripts import active_label

    model = tmp_path / "model.json"
    weights = {feature: 0.0 for feature in FEATURES}
    weights["heart_rate"] = 0.05
    model.write_text(
        json.dumps(
            {
                "model_version": "al-dup-v1",
                "target_id": "deterioration_24h",
                "feature_names": list(FEATURES),
                "weights": weights,
                "bias": -5.0,
                "thresholds": {"medium": 0.2, "high": 0.45, "critical": 0.75},
            }
        ),
        encoding="utf-8",
    )
    inp = tmp_path / "in.csv"
    inp.write_text(
        "encounter_id,patient_id,heart_rate,score,uncertainty\n"
        "e-1,p-1,80,0.9,0.4\n"
        "e-2,p-2,82,0.2,0.3\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "active_label.py",
            "--input",
            str(inp),
            "--model",
            str(model),
            "--top",
            "2",
            "--output",
            str(out),
        ],
    )
    active_label.main()
    header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header.count("score") == 1
    assert header.count("uncertainty") == 1
    assert len(header) == len(set(header))
