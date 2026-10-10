def test_threshold_ladder_reproduces_hand_computed_cutpoints():
    import pytest

    from scripts.thresholds import derive_thresholds

    # 手算（Ruling R6, target=1.0 → 阶梯 TPR 1.0/0.75/0.5；期望值经复核修正，
    # 计划草稿的 0.25/0.35 算错一位，见台账 Task 2 Ruling）：
    # 正例 8 个 0.10..0.45、负例 8 个 0.60..0.95（两簇分离）
    # medium：TPR≥1.0 的最高切点 = 0.10（t=0.15 → 7/8 <1）
    # high：TPR≥0.75 的最高切点 = 0.20（6/8=0.75；t=0.25 → 5/8 <0.75）
    # critical：TPR≥0.5 的最高切点 = 0.30（4/8=0.5；t=0.35 → 3/8 <0.5）
    labels = [1] * 8 + [0] * 8
    scores = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45] + [
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
    ]
    result = derive_thresholds(scores, labels, target_sensitivity=1.0)
    assert result["medium"] == pytest.approx(0.10, abs=1e-9)
    assert result["high"] == pytest.approx(0.20, abs=1e-9)
    assert result["critical"] == pytest.approx(0.30, abs=1e-9)
    assert result["medium"] < result["high"] < result["critical"]


def test_threshold_ladder_rejects_impossible_ordering():
    import pytest

    from scripts.thresholds import derive_thresholds

    # 所有分数相同 → 无法形成严格升序的三级
    with pytest.raises(ValueError, match="threshold"):
        derive_thresholds([0.5] * 10, [1, 0] * 5, target_sensitivity=0.9)


def _gate_like_artifact(path, target_id: str, version: str) -> None:
    import json

    from app.synthetic import FEATURES

    weights = {feature: 0.0 for feature in FEATURES}
    weights["creatinine"] = 1.5
    path.write_text(
        json.dumps(
            {
                "model_version": version,
                "target_id": target_id,
                "feature_names": FEATURES,
                "weights": weights,
                "bias": -3.0,
                "thresholds": {"medium": 0.2, "high": 0.45, "critical": 0.75},
            }
        ),
        encoding="utf-8",
    )


def _tiny_holdout(path) -> None:
    # 8 个不同肌酐值 → 模型分值8 档，足够构造严格升序的三级阈值。
    lines = ["creatinine,label"]
    for creatinine in (5.0, 4.0, 3.0, 2.5, 1.0, 0.5, 0.2, 0.1):
        lines.append(f"{creatinine},{1 if creatinine > 2 else 0}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_thresholds_cli_rejects_target_mismatch_with_clear_error(tmp_path):
    """Review I1：--target-id 与工件不符必须响亮拒绝，禁止 dev 回落打分后回写。"""
    import sys

    import pytest

    from scripts.thresholds import main as thresholds_main

    artifact = tmp_path / "model.json"
    holdout = tmp_path / "holdout.csv"
    _gate_like_artifact(artifact, "deterioration_24h", "mis-v1")
    _tiny_holdout(holdout)
    before = artifact.read_text(encoding="utf-8")
    old_argv = sys.argv
    sys.argv = [
        "thresholds.py",
        "--artifact",
        str(artifact),
        "--input",
        str(holdout),
        "--target-id",
        "readmission_30d",
    ]
    try:
        with pytest.raises(SystemExit) as excinfo:
            thresholds_main()
    finally:
        sys.argv = old_argv
    # 拒绝语来自目标校验（含 "target"），而非 argparse 的 code=2
    assert "target" in str(excinfo.value)
    # 工件未被改写
    assert artifact.read_text(encoding="utf-8") == before


def test_thresholds_cli_derives_for_readmission_artifact(tmp_path):
    """Review I1：--target-id readmission 必须加载给定工件而非默认路径。"""
    import json
    import sys

    from scripts.thresholds import main as thresholds_main

    artifact = tmp_path / "rm.json"
    holdout = tmp_path / "holdout.csv"
    _gate_like_artifact(artifact, "readmission_30d", "rm-t-v1")
    _tiny_holdout(holdout)
    old_argv = sys.argv
    sys.argv = [
        "thresholds.py",
        "--artifact",
        str(artifact),
        "--input",
        str(holdout),
        "--target-id",
        "readmission_30d",
    ]
    try:
        thresholds_main()
    finally:
        sys.argv = old_argv
    written = json.loads(artifact.read_text(encoding="utf-8"))
    thresholds = written["thresholds"]
    assert 0 < thresholds["medium"] < thresholds["high"] < thresholds["critical"] < 1
    # 用的是给定工件（version 不变、阈值已被重写为分位切点）
    assert written["model_version"] == "rm-t-v1"
