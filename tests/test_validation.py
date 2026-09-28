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
