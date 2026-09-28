from scripts.evaluate_model import binary_metrics


def test_binary_metrics_rank_and_threshold_outputs():
    result = binary_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 0.5)
    assert result["auroc"] == 1.0
    assert result["auprc"] == 1.0
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0


def test_news2_score_matches_manual_example():
    from scripts.baselines import news2_score

    # 官方权重手算：RR 25→3、SpO2 92→2、SBP 95→2、HR 125→2、T 39.5→2，合计 11 → high
    score, band = news2_score(
        {
            "respiratory_rate": 25,
            "spo2": 92,
            "systolic_bp": 95,
            "heart_rate": 125,
            "temperature": 39.5,
        }
    )
    assert score == 11
    assert band == "high"


def test_labeled_csv_three_way_comparison(tmp_path):
    import json

    from app.config import Settings
    from app.risk import RiskEngine
    from app.synthetic import FEATURES
    from scripts.evaluate_model import comparison_metrics

    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "model_version": "cmp-v1",
                "feature_names": FEATURES,
                "weights": {feature: 0.0 for feature in FEATURES},
                "bias": 0.5,
                "thresholds": {"medium": 0.2, "high": 0.45, "critical": 0.75},
            }
        ),
        encoding="utf-8",
    )
    engine = RiskEngine(Settings(model_path=str(path)))
    rows = [
        {
            "label": "1",
            "heart_rate": "130",
            "respiratory_rate": "28",
            "systolic_bp": "92",
            "spo2": "91",
            "temperature": "39.2",
            "news2_score": "12",
        },
        {
            "label": "0",
            "heart_rate": "72",
            "respiratory_rate": "16",
            "systolic_bp": "124",
            "spo2": "98",
            "temperature": "36.6",
            "news2_score": "1",
        },
        {"label": "1", "heart_rate": "118", "news2_score": "7"},
        {"label": "0", "heart_rate": "70", "news2_score": "0"},
        {"label": "1", "heart_rate": "135", "news2_score": "13"},
        {"label": "0", "heart_rate": "66", "news2_score": "2"},
    ]
    report = comparison_metrics(rows, engine, "label")
    assert set(report) == {"model", "news2", "model_with_news2"}
    for metrics in report.values():
        assert {"auroc", "auprc", "brier"} <= set(metrics)
