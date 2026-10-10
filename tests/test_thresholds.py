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
