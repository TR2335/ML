import contextlib
import json
import sys
from pathlib import Path


def _train_lgbm(tmp_path, rows: str = "800") -> str:
    from scripts.train_model import main as train_main

    output = str(tmp_path / "lgbm.json")
    old_argv = sys.argv
    sys.argv = [
        "train_model.py",
        "--output",
        output,
        "--rows",
        rows,
        "--version",
        "p7-lgbm-v1",
        "--algorithm",
        "lightgbm",
        "--trials",
        "2",
    ]
    try:
        with contextlib.redirect_stdout(open(str(tmp_path / "train.log"), "w", encoding="utf-8")):
            train_main()
    finally:
        sys.argv = old_argv
    return output


def test_lightgbm_artifact_shape_and_loads(tmp_path):
    from app.config import Settings
    from app.domain import ClinicalEvent
    from app.risk import RiskEngine

    path = _train_lgbm(tmp_path)
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    assert artifact["algorithm"] == "lightgbm"
    assert isinstance(artifact["trees"], list) and artifact["trees"]
    first_node = artifact["trees"][0]["nodes"][0]
    assert {"feature_order_index", "threshold", "left", "right", "value", "default_left"} <= set(
        first_node
    )
    assert artifact["preprocess"] is None
    assert artifact["bias"] == 0.0
    assert all(value == 0.0 for value in artifact["weights"].values())
    assert "drift_reference" in artifact
    assert "booster_text" in artifact
    engine = RiskEngine(Settings(model_path=path))
    assert engine.algorithm == "lightgbm"
    score = engine.score(
        ClinicalEvent(
            event_type="vital_signs",
            patient_id="p",
            encounter_id="e",
            payload={"age": 70, "heart_rate": 120, "creatinine": 2.5},
        )
    )
    assert 0.0 <= score.score <= 1.0
    assert score.model_version == "p7-lgbm-v1"


def test_tree_artifact_scores_match_lightgbm(tmp_path):
    """纯 Python 走树与 LightGBM booster.predict 一致（1e-6，Review Focus #1）。"""
    import math

    import numpy as np
    from lightgbm import Booster

    from app.config import Settings
    from app.domain import ClinicalEvent
    from app.risk import RiskEngine, _isotonic_apply
    from scripts.train_model import build_demo_dataset, feature_names_for

    path = _train_lgbm(tmp_path)
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    booster = Booster(model_str=artifact["booster_text"])
    _times, matrix, _labels = build_demo_dataset("deterioration_24h", 800)
    names = feature_names_for("deterioration_24h")
    x = np.asarray(matrix, dtype=float)
    # serving 分数 = 校准后概率：isotonic 直接输出概率；platt 走 logit 空间
    # （Ruling R-P7-5 语义；deterioration 默认可能触发 R-P7-5a 的 platt 回退）。
    calibration = artifact["calibration"]
    margins = booster.predict(x, raw_score=True)
    if calibration["method"] == "isotonic":
        raw_pred = [min(max(_isotonic_apply(calibration, float(m)), 0.0), 1.0) for m in margins]
    else:
        raw_pred = [
            1 / (1 + math.exp(-(calibration["a"] * float(m) + calibration["b"]))) for m in margins
        ]
    engine = RiskEngine(Settings(model_path=path))
    step = max(1, len(matrix) // 40)
    for i in list(range(0, len(matrix), step))[:40]:
        payload = {}
        for j, name in enumerate(names):
            if name.endswith("__missing"):
                continue
            value = matrix[i][j]
            if not math.isnan(value):
                payload[name] = value
        event = ClinicalEvent(
            event_type="vital_signs",
            patient_id=f"t-p-{i}",
            encounter_id=f"t-e-{i}",
            payload=payload,
        )
        served = engine.score(event).score
        # RiskScore 契约为 4 位舍入（最大偏差 5e-5）；未舍入口径的 1e-6 严格
        # 对照由 Task 2 的归因守恒断言承担（台账 Task 1 Ruling）。
        assert abs(served - float(raw_pred[i])) < 5.1e-5, (i, served, raw_pred[i])


def test_monotone_constraint_holds_on_spo2_ramp(tmp_path):
    """约束生效：SpO2 递增（其余固定）分数不升（WBS T7.1 验收）。"""
    from app.config import Settings
    from app.domain import ClinicalEvent
    from app.risk import RiskEngine

    path = _train_lgbm(tmp_path)
    engine = RiskEngine(Settings(model_path=path))
    scores = []
    for spo2 in (90, 93, 96, 99):
        event = ClinicalEvent(
            event_type="vital_signs",
            patient_id="ramp",
            encounter_id=f"e-{spo2}",
            payload={
                "age": 70,
                "heart_rate": 110,
                "respiratory_rate": 22,
                "systolic_bp": 110,
                "temperature": 37.5,
                "spo2": spo2,
                "wbc": 10,
                "creatinine": 1.5,
                "lactate": 2.0,
                "prior_admissions_30d": 1,
            },
        )
        scores.append(engine.score(event).score)
    from itertools import pairwise

    assert all(a >= b - 1e-9 for a, b in pairwise(scores)), scores


def test_tree_path_contributions_sum_to_logit(tmp_path):
    """归因守恒：完整路径归因和+树基线+bias == 校准前 logit（严格 1e-6，WBS T7.2）。"""
    import math

    from app.config import Settings
    from app.domain import ClinicalEvent
    from app.risk import RiskEngine

    path = _train_lgbm(tmp_path)
    engine = RiskEngine(Settings(model_path=path))
    event = ClinicalEvent(
        event_type="vital_signs",
        patient_id="attr",
        encounter_id="attr-1",
        payload={"age": 74, "heart_rate": 125, "creatinine": 2.8, "lactate": 3.1},
    )
    score = engine.score(event)
    assert 0.0 < score.score < 1.0
    debug = engine.tree_debug(event)
    # 归因守恒（未舍入口径，严格 1e-6）
    assert abs(debug["attribution_sum"] - debug["pre_logit"]) < 1e-6
    # 校准后 logit 与 serving 4 位舍入分数一致（舍入契约 2e-4）
    assert abs(1 / (1 + math.exp(-debug["post_logit"])) - score.score) < 2e-4
    # 归因确实有信号：top_factors 非空且带中文/原名标签
    assert score.top_factors
    assert all(factor.feature for factor in score.top_factors)
    # P6 deferred 5：展示标签不得出现 __missing 裸工程名（树路径同样经 factor_label）。
    assert all("__missing" not in factor.label for factor in score.top_factors)


def test_lgbm_search_space_pairs_subsample_with_freq():
    """P7 deferred 2：subsample 必须配 subsample_freq>0，否则 LightGBM 忽略 bagging，
    整个搜索维度无效；suggest 出的 freq 会自然进入 study.best_params → 最终模型同样生效。"""

    class _StubTrial:
        def __init__(self):
            self.names = []

        def suggest_int(self, name, low, high, **kwargs):
            self.names.append(name)
            return low

        def suggest_float(self, name, low, high, log=False, **kwargs):
            self.names.append(name)
            return low

    from scripts.train_model import _lgbm_search_space

    trial = _StubTrial()
    space = _lgbm_search_space(trial)
    assert "subsample" in space and "subsample_freq" in space
    assert space["subsample_freq"] >= 1  # LightGBM 默认 0 = 不启用 bagging
    assert "subsample_freq" in trial.names  # 经 suggest → 进 study.best_params


def test_tree_engine_rejects_corrupt_trees_block(tmp_path):
    """坏 trees 块响亮拒绝，不静默回落打分（Review 口径）。"""
    import pytest

    from app.config import Settings
    from app.risk import RiskEngine

    path = _train_lgbm(tmp_path)
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    artifact["trees"] = [{"nodes": [{"bogus": 1}]}]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError)):
        RiskEngine(Settings(model_path=str(bad), app_env="test", model_signature_required=False))


def _strict_engine(tmp_path, mutate) -> None:
    """构造被篡改的 lightgbm 工件并以 strict 模式加载——期望响亮拒绝。"""
    import pytest

    from app.config import Settings
    from app.risk import RiskEngine

    path = _train_lgbm(tmp_path)
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    mutate(artifact)
    bad = tmp_path / "mutated.json"
    bad.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError)):
        RiskEngine(Settings(model_path=str(bad), app_env="test", model_signature_required=False))


def test_lightgbm_without_trees_rejected(tmp_path):
    """Review I1：algorithm=lightgbm 但缺 trees 不得静默恒分（0 权重线性路径）。"""

    def drop_trees(artifact):
        artifact["trees"] = None

    _strict_engine(tmp_path, drop_trees)


def test_unknown_algorithm_rejected(tmp_path):
    """Review I1：未知 algorithm 字符串响亮拒绝，不按默认线性路径打分。"""

    def bogus_algorithm(artifact):
        artifact["algorithm"] = "gbdt_magic"

    _strict_engine(tmp_path, bogus_algorithm)


def test_tree_child_index_out_of_bounds_rejected(tmp_path):
    """Review I2：子索引越界在载入期拒绝（否则走树时 IndexError 被兜底成静默降级）。"""

    def oob_child(artifact):
        for tree in artifact["trees"]:
            for node in tree["nodes"]:
                if not node["leaf"]:
                    node["left"] = 10**6
                    return

    _strict_engine(tmp_path, oob_child)


def test_tree_internal_negative_feature_index_rejected(tmp_path):
    """Review I2：内部节点 feature_order_index=-1 会经 Python 负索引读到最后一个特征——静默错分。"""

    def negative_index(artifact):
        for tree in artifact["trees"]:
            for node in tree["nodes"]:
                if not node["leaf"]:
                    node["feature_order_index"] = -1
                    return

    _strict_engine(tmp_path, negative_index)


def test_tree_cycle_rejected(tmp_path):
    """Review I2：环形 left/right 会让 serving 走树无限挂起——拍平序前序性保证 child>parent。"""

    def make_cycle(artifact):
        tree = artifact["trees"][0]
        root = tree["nodes"][0]
        if root["leaf"]:
            root["leaf"] = False
        # 根指向自己成环（child == parent 违反前序递增）
        root["left"] = 0
        root["right"] = 0
        root["feature_order_index"] = 0
        root["threshold"] = 0.0

    _strict_engine(tmp_path, make_cycle)


def test_unrounded_tree_margin_matches_booster(tmp_path):
    """Review I3：未舍入口径 tree_raw_margin == booster.predict(raw_score=True)，严格 1e-6。"""
    import math

    import numpy as np
    from lightgbm import Booster

    from app.config import Settings
    from app.domain import ClinicalEvent
    from app.risk import RiskEngine
    from scripts.train_model import build_demo_dataset, feature_names_for

    path = _train_lgbm(tmp_path)
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    booster = Booster(model_str=artifact["booster_text"])
    _times, matrix, _labels = build_demo_dataset("deterioration_24h", 800)
    names = feature_names_for("deterioration_24h")
    engine = RiskEngine(Settings(model_path=path))
    step = max(1, len(matrix) // 40)
    for i in list(range(0, len(matrix), step))[:20]:
        payload = {}
        for j, name in enumerate(names):
            if name.endswith("__missing"):
                continue
            value = matrix[i][j]
            if not math.isnan(value):
                payload[name] = value
        event = ClinicalEvent(
            event_type="vital_signs",
            patient_id=f"m-p-{i}",
            encounter_id=f"m-e-{i}",
            payload=payload,
        )
        expected = float(booster.predict(np.asarray([matrix[i]], dtype=float), raw_score=True)[0])
        assert abs(engine.tree_raw_margin(event) - expected) < 1e-6, (i, expected)
