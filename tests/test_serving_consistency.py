import csv
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TARGETS = ["deterioration_24h", "readmission_30d"]


def _load_fixture(target_id: str):
    directory = FIXTURES / target_id
    with (directory / "golden_dataset.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    artifact = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    return rows, artifact


def _payload_from_row(row: dict) -> dict[str, float]:
    from app.synthetic import FEATURES

    payload = {
        feature: float(row[feature])
        for feature in FEATURES
        if row.get(feature, "") not in ("", None)
    }
    if row.get("news2_score", "") not in ("", None):
        payload["news2_score"] = float(row["news2_score"])
    return payload


@pytest.mark.parametrize("target_id", TARGETS)
def test_offline_online_scores_match(target_id, tmp_path):
    from app.config import Settings
    from app.domain import ClinicalEvent
    from app.risk import RiskEngine
    from scripts.evaluate_model import evaluate_rows

    rows, artifact = _load_fixture(target_id)
    assert len(rows) == 200
    artifact_path = tmp_path / "model.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    engine = RiskEngine(Settings(model_path=str(artifact_path), app_env="dev"), target_id=target_id)
    offline_labels, offline_scores = evaluate_rows(rows, engine, "label")
    serving_scores = [
        engine.score(
            ClinicalEvent(
                event_type="vital_signs",
                patient_id=str(row.get("patient_id", f"g-p-{i}")),
                encounter_id=str(row.get("encounter_id", f"g-e-{i}")),
                payload=_payload_from_row(row),
            )
        ).score
        for i, row in enumerate(rows)
    ]
    for i in range(len(rows)):
        assert abs(offline_scores[i] - serving_scores[i]) < 1e-12  # 双路径同分
        assert abs(serving_scores[i] - float(rows[i]["serving_score"])) < 1e-9  # 与冻结值一致
        assert abs(offline_scores[i] - float(rows[i]["offline_score"])) < 1e-9
        assert int(rows[i]["label"]) == offline_labels[i]


def test_golden_catches_contract_drift(tmp_path):
    """篡改工件 preprocess mean → 分数与冻结值背离 → 一致性契约变红。

    fixture 行零缺失，median 填补路径不触发（见台账 Task 5 Ruling），故篡改
    每行必用的中心化 mean。
    """
    from app.config import Settings
    from app.domain import ClinicalEvent
    from app.risk import RiskEngine

    rows, artifact = _load_fixture("deterioration_24h")
    tampered = json.loads(json.dumps(artifact))
    tampered["preprocess"]["heart_rate"]["mean"] = (
        float(tampered["preprocess"]["heart_rate"]["mean"]) + 5.0
    )
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    engine = RiskEngine(Settings(model_path=str(tampered_path), app_env="dev"))
    drifted = 0
    for i, row in enumerate(rows):
        score = engine.score(
            ClinicalEvent(
                event_type="vital_signs",
                patient_id=f"p{i}",
                encounter_id=f"e{i}",
                payload=_payload_from_row(row),
            )
        ).score
        if abs(score - float(row["serving_score"])) > 1e-9:
            drifted += 1
    assert drifted > 0, "篡改 preprocess 后必须出现分数漂移（fixture 一致性测试才会变红）"
