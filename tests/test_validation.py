def _timed_rows(count: int = 40) -> list[dict]:
    from datetime import UTC, datetime, timedelta

    base = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        {
            "encounter_id": f"enc-{index:03d}",
            "index_time": (base + timedelta(hours=index)).isoformat(),
            "label": str(index % 2),
        }
        for index in range(count)
    ]


def test_rolling_time_split_ordered_and_disjoint():
    from scripts.validate import rolling_time_split

    rows = list(reversed(_timed_rows()))  # 输入故意乱序
    folds = rolling_time_split(rows, n_splits=4)
    assert len(folds) == 4
    seen_test_ids: set[str] = set()
    for train, test in folds:
        train_times = sorted(row["index_time"] for row in train)
        test_times = sorted(row["index_time"] for row in test)
        assert train_times[-1] < test_times[0]  # train 全部早于 test（无泄漏）
        test_ids = {row["encounter_id"] for row in test}
        assert test_ids.isdisjoint(seen_test_ids)  # test 块互不重叠
        seen_test_ids |= test_ids
    # 首块只做初始 train：40 行分 5 块 × 8 行，test 合计 4 块
    assert len(seen_test_ids) == 32


def test_bootstrap_ci_coverage_sane():
    from random import Random

    from scripts.validate import bootstrap_metric

    def mean_metric(labels, scores):
        return sum(labels) / len(labels)

    # 点估计 == 样本均值（确定性）
    labels = [1, 0, 1, 1, 0]
    point, lo, hi = bootstrap_metric(labels, labels, mean_metric, n=200, seed=7)
    assert point == 0.6
    assert lo <= point <= hi
    # 覆盖率：对已知总体 p=0.3 做 100 次「抽样 + bootstrap」，95% 名义覆盖 ≥85%
    covered = 0
    reps = 100
    for rep in range(reps):
        data_rng = Random(rep)
        sample = [1 if data_rng.random() < 0.3 else 0 for _ in range(200)]
        _, lo_r, hi_r = bootstrap_metric(sample, sample, mean_metric, n=200, seed=rep + 10_000)
        covered += int(lo_r <= 0.3 <= hi_r)
    assert covered >= 85, f"覆盖率 {covered}/{reps} 过低"


def test_rolling_time_split_keeps_duplicate_timestamps_on_one_side():
    from datetime import UTC, datetime, timedelta

    from scripts.validate import rolling_time_split

    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for tick in range(50):  # 50 个时刻 × 每刻 3 行（真实数据常见同刻多 landmark）
        for copy in range(3):
            rows.append(
                {
                    "encounter_id": f"enc-{tick:03d}-{copy}",
                    "index_time": (base + timedelta(hours=tick)).isoformat(),
                    "label": str((tick + copy) % 2),
                }
            )
    folds = rolling_time_split(rows, n_splits=3)
    assert len(folds) == 3
    for train, test in folds:
        train_max = max(row["index_time"] for row in train)
        test_min = min(row["index_time"] for row in test)
        # 同刻行必须整体落同一侧：train 严格早于 test（不允许 TIE）
        assert train_max < test_min, f"边界 TIE: train_max={train_max} test_min={test_min}"


def test_subgroup_sensitivity_diff_flags_imbalance():
    from scripts.validate import subgroup_sensitivity_diff

    # 老组 10 阳性全检出（sens=1.0）；年轻组 10 阳性只检出 2（sens=0.2）→ 差 0.8
    labels = [1] * 20
    scores = [0.9] * 10 + [0.9, 0.9] + [0.1] * 8
    groups = ["older"] * 10 + ["younger"] * 10
    diff = subgroup_sensitivity_diff(labels, scores, groups, threshold=0.5, min_positives=10)
    assert diff is not None and abs(diff - 0.8) < 1e-9
    # 前置不足 → None（降级跳过）
    assert (
        subgroup_sensitivity_diff([1] * 5, [0.9] * 5, ["older"] * 5, 0.5, min_positives=10) is None
    )
    assert (
        subgroup_sensitivity_diff([1] * 20, [0.9] * 20, ["same"] * 20, 0.5, min_positives=10)
        is None  # 单组无从比较
    )


def test_decision_curve_mean_gain_both_sides():
    from scripts.validate import decision_curve_mean_gain

    labels = [1, 1, 0, 0]
    perfect = [0.9, 0.8, 0.2, 0.1]
    inverted = [0.1, 0.2, 0.8, 0.9]
    assert decision_curve_mean_gain(labels, perfect) > 0
    assert decision_curve_mean_gain(labels, inverted) < 0


def test_per_facility_auprc_requires_two_facilities():
    from scripts.validate import per_facility_auprc

    rows = [{"facility_id": "h1"}] * 30
    labels = [1] * 10 + [0] * 20
    scores = [0.9] * 10 + [0.1] * 20
    single = per_facility_auprc(rows, labels, scores, baseline_auprc=0.3)
    assert single["checked"] is False
    rows2 = [{"facility_id": "h1" if i < 30 else "h2"} for i in range(60)]
    labels2 = labels + labels
    both = per_facility_auprc(rows2, labels2, scores + scores, baseline_auprc=0.3)
    assert both["checked"] is True and not both["failures"]
    # h2 反向排序：10 阳性压到 20 阴性之后 → AUPRC≈0.21 < 基线 0.3 → 拒
    inverted2 = scores + [0.05] * 10 + [0.9] * 20
    bad = per_facility_auprc(rows2, labels2, inverted2, baseline_auprc=0.3)
    assert bad["checked"] is True and bad["failures"]
    # 缺 facility 列 → 降级跳过
    missing = per_facility_auprc([{}] * 60, labels2, scores + scores, baseline_auprc=0.3)
    assert missing["checked"] is False
