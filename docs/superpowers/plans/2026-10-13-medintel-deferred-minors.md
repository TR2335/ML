# MedIntel 24 项终审 Deferred Minors 清偿实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 一次性清偿 P6/P7/P8 三轮终审留下的全部 24 项 deferred minor（P6×10、P7×8、P8×6），不引入行为回归。

**Architecture:** 按评审轮次分三个阶段共 21 个任务。每任务 = failing test → 最小修复 → 全量回归 → 独立提交。互相关联的项合并为单任务（P6-7+P7-1 门禁输入、P6-1+P6-10 工件口径、P7-5+P7-8 recalibrate 标签、P7-4+P7-7 架构文档），避免同文件冲突。全程 TDD：先写测试看它失败，再改实现。

**Tech Stack:** Python 3.11+、pytest、ruff、scikit-learn（仅训练/评估依赖，serving 保持纯标准库）。

**Spec:** 问题清单原文在 `docs/superpowers/STATUS.md`「终审 Deferred Minors（24 项，未修）」节（本计划按任务重新编号映射）；探查锚点来自各轮评审计划 `docs/superpowers/plans/2026-09-28-medintel-p6.md`、`2026-10-10-medintel-p7.md`、`2026-10-10-medintel-p8.md`。

## Global Constraints

- 基线（2026-10-13 实测）：`pytest` → **196 passed + 1 skipped + 2 deselected**；`ruff check .` 全零。每任务结束两者必须保持全绿。
- `app/` serving 层保持纯标准库（不得引入 scikit-learn 等训练依赖）。
- TDD：每个修复先落测试、确认失败，再改实现。禁止 `skip`/`only`/TODO 占位。
- 每任务独立提交，风格沿用仓库既有前缀（`fix:`/`test:`/`docs:`/`chore:`）。
- 涉及工件 schema 的变更（Task 1）必须重冻结 golden fixtures 并过 `make_golden --check`。
- 运行命令统一用 `.venv/Scripts/python -m pytest` / `.venv/Scripts/python scripts/xxx.py`（Windows）。
- 全套测试约 10 分钟（含 1500 规模 `@slow` 冒烟）：每任务收尾跑全套；时间敏感时至少跑受影响文件 + `pytest -q -m "not slow"`，最后任务跑全套。

## Review Focus

1. **P6-9 平局 AUPRC 修复**：门禁 fixture `_write_gate_rows`（tests/test_manage_model.py:6-18）的 news2 baseline 恒定 → 修复后 baseline_auprc 从 1.0 变 prevalence(0.25)；模型分数由 creatinine 两点定线（3.0→高分、0.8→低分，标签同源）→ 模型 AP 仍为 1.0，断言 `auprc_ci.lo >= baseline_auprc` 应保持通过；`invert_labels` 拒绝路径 baseline 变低后需确认仍拒绝。跑 `tests/test_manage_model.py` + `tests/test_production_controls.py` 验证，若拒绝路径失效则收紧 fixture（见 Task 9 Step 5 应急步骤）。
2. **P6-2 阈值撞档改拒绝**：`derive_thresholds` 对撞档从「静默抬高」改为「响亮拒绝」；若 demo 训练/thresholds CLI 的分数分布撞档，训练路径会 raise。全套（含 `@slow` 冒烟）确认既有分布不撞档；若撞档说明原行为已在产出不达 TPR 的档位——按 Task 9 应急步骤处置。
3. **Task 1 工件 schema 变更**（training_rows 语义 + 校准审计字段）：`tests/fixtures/**/model.json` 须重冻结；CI 的 `make_golden --check` 与 `tests/test_serving_consistency.py` 必须绿。
4. **P7-5/8 recalibrate 拒绝语来源改变**：全零 gate-input 的 SystemExit 文案从门禁的 "holdout needs at least one positive" 变为 "recalibrate: gate-input needs both classes"；`tests/test_drift.py:140-169` 的期望若 match 旧文案必须同步改。门禁对其他调用方的兜底检查保留不动。
5. **P8-4 双出院钉首**：仅承诺顺序到达场景（upsert 防倒退使乱序到达仍停留较晚 as_of）；`docs/data-contract.md` 措辞不得对乱序做超出实现的承诺；第二次出院的 deterioration 重打分不受影响。

## 项→任务映射

| 原始项 | 任务 |
|---|---|
| P6-1 training_rows、P6-10 校准审计 | Task 1 |
| P6-2 阈值撞档 | Task 2 |
| P6-3 阈值常量双份 | Task 3 |
| P6-4 CV 零阳性折 | Task 4 |
| P6-5 `__missing` 标签 | Task 5 |
| P6-6 active_label 重复列 | Task 6 |
| P6-7 门禁统一拒绝语 + P7-1 age 裸异常/措辞 | Task 7 |
| P6-8 make_golden 路径耦合 | Task 8 |
| P6-9 恒分 AUPRC | Task 9 |
| P7-2 Optuna subsample 惰性 | Task 10 |
| P7-3 facility 全跳过 / 亚组前两组 | Task 11 |
| P7-5 gate-input 预检 + P7-8 int("2") | Task 12 |
| P7-6 isotonic Brier 断言 | Task 13 |
| P7-4 drift schema 文档 + P7-5a 回退文档 | Task 14 |
| P8-1 死常量 | Task 15 |
| P8-2 summary 漏字段 | Task 16 |
| P8-3 白名单半边回归测试 | Task 17 |
| P8-4 多出院语义 | Task 18 |
| P8-5 双常量交叉测试 | Task 19 |
| P8-6 LACE 缺列集成测试 | Task 20 |
| 收尾验证与状态归档 | Task 21 |

---

### Task 1: 工件口径 — training_rows 语义 + 校准审计字段 + golden 重冻结（P6-1、P6-10）

**Files:**
- Modify: `scripts/train_model.py`（:504 切分、:547 与 :577 两处 `"training_rows": len(y)`；:442-470 `_fit_isotonic_block`）
- Modify: `tests/test_training.py`（:364 断言；新增校准审计测试）
- Modify: `tests/test_drift.py`（recalibrate 后校准审计断言，:138 附近）
- Regenerate: `tests/fixtures/deterioration_24h/model.json`、`tests/fixtures/readmission_30d/model.json`

**Interfaces:**
- Produces: 工件 `training_rows` = 实际 fit 的 60% 训练段行数（不再记全量）；`calibration` 块新增 `n_samples: int`、`n_positives: int`（两个分支都有，确定性字段——**不得加时间戳**，否则 golden 不可复现）。后续任务与报告卡不依赖这些字段的旧语义。

- [ ] **Step 1: 改测试期望（红）**

`tests/test_training.py:364`（输入 CSV 恰 40 行的用例）：

```python
    # P6 deferred 1：training_rows 只计实际 fit 的 60% 训练段，不再记全量。
    assert artifact["training_rows"] == 24  # int(40 * 0.6)
```

- [ ] **Step 2: 新增校准审计测试（红）**

`tests/test_training.py` 新增：

```python
def test_calibration_block_carries_sample_audit(tmp_path):
    """P6 deferred 10：校准审计链除 fitted_on 外还须有样本量与阳性数。"""
    path = _train(tmp_path)
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    cal = artifact["calibration"]
    assert cal["n_samples"] > 0
    assert 0 < cal["n_positives"] <= cal["n_samples"]
    if cal["fitted_on"] == "train_segment":
        # 校准拟合自训练段 → 样本数与 training_rows 同源（两项修复互相钉住）。
        assert cal["n_samples"] == artifact["training_rows"]
```

- [ ] **Step 3: 跑测试确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_training.py -q -k "training_rows or sample_audit or rows"`（若 Step 1 改的是已有用例名，直接跑该用例）
Expected: FAIL（40≠24；缺 `n_samples` 键 KeyError）

- [ ] **Step 4: 实现**

`scripts/train_model.py`：
1. :547 与 :577 两处（先读上下文确认 `train_idx` 在作用域内、变量名一致）：

```python
            "training_rows": len(train_idx),  # 只记实际 fit 的训练段（P6 deferred 1）
```

2. `_fit_isotonic_block`（:442-470）两个 return 分支都补审计字段（在 `fitted_on` 之后）：

```python
        "n_samples": len(labels),
        "n_positives": int(sum(labels)),
```

`scripts/recalibrate.py:88` 传入的 `labels` 经过 `_fit_isotonic_block` 内部自算——**无需改签名**。

- [ ] **Step 5: 补 recalibrate 审计断言（绿）**

`tests/test_drift.py` recalibrate 测试（:138 断言 `fitted_on == "recalibration_set"` 旁）新增：

```python
    # P6 deferred 10：再校准块同样携带样本审计。
    block = json.loads(artifact_path.read_text(encoding="utf-8"))["calibration"]
    recal_rows = list(csv.DictReader(Path(<recal输入csv>).open(encoding="utf-8")))
    assert block["n_samples"] == len(recal_rows)
    assert block["n_positives"] == sum(1 for r in recal_rows if r["label"] == "1")
```

（`<recal输入csv>` = 该测试已有的输入路径变量名；若测试已读过 rows 则直接用其长度。）

- [ ] **Step 6: 重冻结 golden fixtures**

```bash
.venv/Scripts/python scripts/train_model.py --output data/model.json --target-id deterioration_24h
.venv/Scripts/python scripts/train_model.py --output data/readmission_model.json --target-id readmission_30d
.venv/Scripts/python scripts/make_golden.py
.venv/Scripts/python scripts/make_golden.py --check
```

Expected: `--check` 无差异退出 0。fixture 中 `training_rows` 与 `calibration` 字段随之更新。

- [ ] **Step 7: 全套回归**

Run: `.venv/Scripts/python -m pytest -q` + `.venv/Scripts/python -m ruff check .`
Expected: 196+ 新增测试全绿、ruff 零（若 `test_serving_consistency` 报 fixture 分数差异 → 重跑 Step 6 确认 `--check` 通过；分数本身不应因本任务改变——training_rows/审计字段不参与打分）。

- [ ] **Step 8: Commit**

```bash
git add scripts/train_model.py tests/test_training.py tests/test_drift.py tests/fixtures data
git commit -m "fix: training_rows counts fit segment and calibration block carries sample audit (P6 deferred 1/10)"
```

---

### Task 2: derive_thresholds 撞档响亮拒绝（P6-2）

**Files:**
- Modify: `scripts/thresholds.py:44-54`
- Test: `tests/test_thresholds.py`

**Interfaces:**
- Produces: `derive_thresholds(scores, labels, target_sensitivity=0.90) -> dict[str, float]` 签名不变；撞档时 raise `ValueError("threshold: cannot form strictly ascending cutpoints from this score distribution")`（与既有全同分拒绝同一文案，既有测试不受影响）。

**关键推理**（执行者须理解，勿走回头路）：`_max_threshold_at_tpr` 返回满足目标 TPR 的**最大**切点。若某档候选 ≤ 上一档，则任何更高切点的 TPR 必然低于该档目标——旧的「抬到下一档」循环产出的档位**永远不达其 TPR 目标**。因此修复不是"抬高后复检"，而是撞档即拒绝。

- [ ] **Step 1: 写失败测试**

`tests/test_thresholds.py` 新增：

```python
def test_derive_thresholds_rejects_collision_that_breaks_tpr():
    """P6 deferred 2：撞档抬高必然跌破该档 TPR 目标 → 必须拒绝而非静默接受。
    positives={0.5,0.5,0.8,0.9}：medium(.9) best=0.5；high(.675) best 也是 0.5
    （下一档 0.6 处 TPR=0.5 < 0.675），旧实现会静默抬到 0.6 产出不达标的 high 档。
    """
    from scripts.thresholds import derive_thresholds
    import pytest

    scores = [0.1, 0.5, 0.5, 0.6, 0.8, 0.9]
    labels = [0, 1, 1, 0, 1, 1]
    with pytest.raises(ValueError, match="strictly ascending cutpoints"):
        derive_thresholds(scores, labels)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_thresholds.py::test_derive_thresholds_rejects_collision_that_breaks_tpr -q`
Expected: FAIL（旧实现返回 `{"medium": 0.5, "high": 0.6, "critical": ...}` 不抛）

- [ ] **Step 3: 实现**

`scripts/thresholds.py:44-54`，整段替换为：

```python
    for name, tpr in ladder:
        candidate = _max_threshold_at_tpr(scores, labels, tpr)
        if candidate <= previous:
            # _max_threshold_at_tpr 已取满足目标的最大切点——更高切点必然跌破
            # 该档 TPR 目标，静默抬高曾产出不达标的档位（P6 deferred 2）。
            raise ValueError(
                "threshold: cannot form strictly ascending cutpoints from this score distribution"
            )
        result[name] = candidate
        previous = candidate
```

- [ ] **Step 4: 跑测试确认通过 + 全套**

Run: `.venv/Scripts/python -m pytest tests/test_thresholds.py -q`，再全套 `-q`
Expected: 全绿（手算 {0.10,0.20,0.30} 用例不撞档；若全套有训练/thresholds 用例撞档报出——停下，在 PR/提交说明中报告该分布，不要回退成静默接受）。

- [ ] **Step 5: Commit**

```bash
git add scripts/thresholds.py tests/test_thresholds.py
git commit -m "fix: reject threshold ladder collisions instead of emitting sub-target cutpoints (P6 deferred 2)"
```

---

### Task 3: 阈值常量单一来源（P6-3）

**Files:**
- Modify: `scripts/train_model.py:39-44`（删 `TARGET_THRESHOLDS`，改用 `app.risk.DEFAULT_THRESHOLDS`）、:549、:581 消费点
- Test: `tests/test_training.py` 或 `tests/test_thresholds.py`

**Interfaces:**
- Produces: `app.risk.DEFAULT_THRESHOLDS: dict[str, dict[str, float]]` 为唯一阈值字面量来源；`scripts.train_model` 不再定义 `TARGET_THRESHOLDS`。

- [ ] **Step 1: 写失败测试**

```python
def test_threshold_defaults_have_single_source():
    """P6 deferred 3：train_model 不得再持有独立的阈值字面量副本。"""
    import scripts.train_model as train_model
    from app.risk import DEFAULT_THRESHOLDS

    assert not hasattr(train_model, "TARGET_THRESHOLDS")
    assert DEFAULT_THRESHOLDS["deterioration_24h"] == {
        "medium": 0.2, "high": 0.45, "critical": 0.75,
    }
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_training.py::test_threshold_defaults_have_single_source -q`
Expected: FAIL（`TARGET_THRESHOLDS` 仍存在）

- [ ] **Step 3: 实现**

1. `scripts/train_model.py` 删除 :39-44 的 `TARGET_THRESHOLDS` 字典定义，在 import 区加 `from app.risk import DEFAULT_THRESHOLDS`（`app.risk` 仅标准库，无循环导入——`app.risk` 不 import `scripts`）。
2. :549、:581 两处 `TARGET_THRESHOLDS[...]` 改为 `DEFAULT_THRESHOLDS[...]`。

- [ ] **Step 4: 跑测试确认通过 + 全套 + ruff**

- [ ] **Step 5: Commit**

```bash
git add scripts/train_model.py tests/test_training.py
git commit -m "fix: single-source band thresholds in app.risk.DEFAULT_THRESHOLDS (P6 deferred 3)"
```

---

### Task 4: CV 零阳性测试折跳过而非计 0（P6-4）

**Files:**
- Modify: `scripts/train_model.py`（logistic CV :247-258、lightgbm CV :415-425，新增 `_record_fold_auprc` 帮助函数）
- Test: `tests/test_training.py`

**Interfaces:**
- Produces: `_record_fold_auprc(fold_scores: list[float], test_labels: list[int], pred) -> None` —— 测试折无阳性（`binary_metrics` 返回 `auprc=None`）时跳过该折；两条 CV 路径共用。既有 `if not scores: continue` / `if not auprcs: continue` 全折跳过分支保持。

- [ ] **Step 1: 写失败测试**

```python
def test_record_fold_auprc_skips_positive_free_fold():
    """P6 deferred 4：测试折无阳性 → 跳过该折，不记 0（记 0 会压低均值扭曲 C 选择）。"""
    from scripts.train_model import _record_fold_auprc

    bucket: list[float] = []
    _record_fold_auprc(bucket, [0, 0], [0.4, 0.6])   # 无阳性 → auprc=None
    assert bucket == []
    _record_fold_auprc(bucket, [0, 1], [0.4, 0.6])
    assert len(bucket) == 1 and 0.0 < bucket[0] <= 1.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_training.py::test_record_fold_auprc_skips_positive_free_fold -q`
Expected: FAIL（ImportError：函数不存在）

- [ ] **Step 3: 实现**

`scripts/train_model.py`（放 `_fit_pipeline` 之前）：

```python
def _record_fold_auprc(fold_scores: list[float], test_labels: list[int], pred) -> None:
    """测试折无阳性时 binary_metrics 返回 auprc=None——跳过该折而不是记 0
    （P6 deferred 4：记 0 会系统性压低均值，扭曲 C/超参选择）。"""
    value = binary_metrics(test_labels, list(map(float, pred)), 0.5)["auprc"]
    if value is not None:
        fold_scores.append(value)
```

1. logistic 路径 :254-256 的 `value = ...` + `scores.append(value or 0.0)` 两行替换为 `_record_fold_auprc(scores, test_labels, pred)`。
2. lightgbm 路径 :421-422 的 `metrics = ...` + `auprcs.append(... or 0.0)` 替换为 `_record_fold_auprc(auprcs, test_labels, pred)`（先读上下文核对变量名）。

- [ ] **Step 4: 跑测试确认通过 + 全套**

Run: `.venv/Scripts/python -m pytest tests/test_training.py tests/test_lightgbm.py -q`，再全套
Expected: 全绿（CV 行为仅在零阳性折上变化；demo 数据 CV 若有此类折，C 选择可能变化——若导致某用例失败，报告该用例并核对其对 C 的期望是否本就依赖 0.0 计入）。

- [ ] **Step 5: Commit**

```bash
git add scripts/train_model.py tests/test_training.py
git commit -m "fix: skip positive-free CV test folds instead of scoring them zero (P6 deferred 4)"
```

---

### Task 5: top_factors 不再暴露 `__missing` 裸工程名（P6-5）

**Files:**
- Modify: `app/risk.py`（新增 `factor_label`；:459 线性路径、:577 树路径改用）
- Test: `tests/test_risk_engine.py`

**Interfaces:**
- Produces: `app.risk.factor_label(name: str) -> str` —— `"<base>__missing"` → `f"{LABELS.get(base, base)}缺失"`；其余走 `LABELS.get(name, name)`。两条打分路径统一调用。

- [ ] **Step 1: 写失败测试**

`tests/test_risk_engine.py` 新增：

```python
def test_factor_label_hides_missing_engineering_names():
    """P6 deferred 5：top_factors 展示标签不得出现 __missing 裸工程名。"""
    from app.risk import LABELS, factor_label

    # 指示列 → 「基特征缺失」（基特征已知则复用其中文名）
    label = factor_label("heart_rate__missing")
    assert label.endswith("缺失") and "__missing" not in label
    assert label == f"{LABELS.get('heart_rate', 'heart_rate')}缺失"
    # 未知基特征也兜底成「名缺失」
    assert factor_label("mystery_feature__missing") == "mystery_feature缺失"
    # 普通特征透传
    assert factor_label("mystery_feature") == "mystery_feature"
    assert factor_label("news2_score") == LABELS.get("news2_score", "news2_score")
```

在既有 top_factors 用例（:48 附近，及 `tests/test_lightgbm.py:171-173`）追加统一断言：

```python
    assert all("__missing" not in factor.label for factor in score.top_factors)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_risk_engine.py -q`
Expected: FAIL（ImportError：`factor_label` 不存在）

- [ ] **Step 3: 实现**

`app/risk.py`（`LABELS` 定义之后）：

```python
MISSING_SUFFIX = "__missing"


def factor_label(name: str) -> str:
    """工程特征名 → 展示标签。`__missing` 指示列展示为「基特征缺失」，
    不把裸工程名透给临床用户（P6 deferred 5）。"""
    if name.endswith(MISSING_SUFFIX):
        base = name[: -len(MISSING_SUFFIX)]
        return f"{LABELS.get(base, base)}缺失"
    return LABELS.get(name, name)
```

:459 `label=LABELS.get(f, f)` 与 :577 `label=LABELS.get(name, name)` 两处替换为 `label=factor_label(...)`。

- [ ] **Step 4: 跑测试确认通过 + 全套**

（`app/` 仍纯标准库 ✓；若 golden/报告卡有 `__missing` 标签的快照——跑全套确认，报告卡 top_factors 展示随之变化属预期。）

- [ ] **Step 5: Commit**

```bash
git add app/risk.py tests/test_risk_engine.py tests/test_lightgbm.py
git commit -m "fix: render missing-indicator factors as 'base feature missing' labels (P6 deferred 5)"
```

---

### Task 6: active_label 输出列去重（P6-6）

**Files:**
- Modify: `scripts/active_label.py:90`
- Test: `tests/test_active_label.py`

**Interfaces:**
- Consumes: 既有 CLI `main()`（参数 `--input --model --target-id --top --output`）、`tests/fixtures/deterioration_24h/model.json` 作为 `--model`。
- Produces: 输出 CSV 表头中 `score`/`uncertainty` 各出现一次（输入已含它们时不再追加重复列）。

- [ ] **Step 1: 写失败测试**

`tests/test_active_label.py` 新增：

```python
def test_active_label_cli_dedupes_score_columns(tmp_path, monkeypatch):
    """P6 deferred 6：输入已含 score/uncertainty（对上轮输出再跑）→ 输出表头不重复。"""
    import sys

    from scripts import active_label

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
            "active_label.py", "--input", str(inp),
            "--model", "tests/fixtures/deterioration_24h/model.json",
            "--top", "2", "--output", str(out),
        ],
    )
    active_label.main()
    header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header.count("score") == 1
    assert header.count("uncertainty") == 1
    assert len(header) == len(set(header))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_active_label.py::test_active_label_cli_dedupes_score_columns -q`
Expected: FAIL（`header.count("score") == 2`）

- [ ] **Step 3: 实现**

`scripts/active_label.py:90`：

```python
    # 输入已含 score/uncertainty（对上轮输出再跑）时不追加重复列（P6 deferred 6）。
    out_fields = list(dict.fromkeys([*fields, "score", "uncertainty"]))
```

- [ ] **Step 4: 跑测试确认通过 + 全套**

（若 `payload_from_row` 对缺列输入报错 → 在输入 CSV 中补齐 `tests/test_active_label.py` 既有用例使用的特征列名，断言不变。）

- [ ] **Step 5: Commit**

```bash
git add scripts/active_label.py tests/test_active_label.py
git commit -m "fix: dedupe score/uncertainty columns in active-label output (P6 deferred 6)"
```

---

### Task 7: 门禁输入错误统一拒绝语 + age 优雅降级（P6-7 + P7-1）

**Files:**
- Modify: `scripts/manage_model.py`（:116 读文件、:126 evaluate_rows、:139-143 news2、:163-175 age）
- Test: `tests/test_manage_model.py`

**Interfaces:**
- Consumes: 既有 harness `_write_gate_artifact(path, version)`、`_write_gate_rows(csv_path, invert_labels)`（tests/test_manage_model.py:6-40）、promote CLI 模式（:399-431）。
- Produces: 门禁对**输入类错误**（文件不存在、缺 label 列、label 非数字、news2/vitals 非数字）一律 `SystemExit("promotion gate rejected: ...")` 且不留 manifest；**age 问题不拒绝**——age 整列缺失 → `skipped: "age column absent"`（既有），age 部分空/非数值 → `skipped: "age column present but values missing/non-numeric"`（不再抛裸 ValueError 崩门禁）。

- [ ] **Step 1: 写失败测试（4 个）**

`tests/test_manage_model.py` 新增（promote 调用照 :399-431 的 argv 模式）：

```python
def test_gate_rejects_input_class_errors_with_unified_message(tmp_path):
    """P6 deferred 7：缺 label 列 / label 非数字 / 文件不存在 → 统一拒绝语，不留 manifest。"""
    import pytest
    from scripts.manage_model import main as manage_main

    artifact = tmp_path / "good.json"
    _write_gate_artifact(artifact, "inerr-v1")
    registry = tmp_path / "registry"

    cases = {
        "missing_label": "encounter_id,creatinine\nenc-0001,3.0\nenc-0002,0.8\n",
        "non_numeric_label": (
            "encounter_id,creatinine,label\n"
            "enc-0001,3.0,x\nenc-0002,0.8,0\n" * 8
        ),
    }
    for name, body in cases.items():
        holdout = tmp_path / f"{name}.csv"
        holdout.write_text(body, encoding="utf-8")
        old_argv = sys.argv
        sys.argv = [
            "manage_model.py", "--registry-dir", str(registry),
            "promote", "--artifact", str(artifact),
            "--target-id", "deterioration_24h", "--input", str(holdout),
        ]
        try:
            with pytest.raises(SystemExit, match="promotion gate rejected"):
                manage_main()
        finally:
            sys.argv = old_argv
        assert not (registry / "manifest.json").exists()

    # 文件不存在
    old_argv = sys.argv
    sys.argv = [
        "manage_model.py", "--registry-dir", str(registry),
        "promote", "--artifact", str(artifact),
        "--target-id", "deterioration_24h", "--input", str(tmp_path / "nope.csv"),
    ]
    try:
        with pytest.raises(SystemExit, match="promotion gate rejected"):
            manage_main()
    finally:
        sys.argv = old_argv
    assert not (registry / "manifest.json").exists()


def test_gate_age_non_numeric_skips_rule_instead_of_crashing(tmp_path):
    """P7 deferred 1：age 非数值/部分空 → subgroup 规则 skipped（措辞区分 absent vs 值缺失），门禁不崩。"""
    import pytest
    from scripts.manage_model import main as manage_main

    artifact = tmp_path / "good.json"
    _write_gate_artifact(artifact, "age-v1")
    holdout = tmp_path / "holdout.csv"
    lines = ["encounter_id,creatinine,heart_rate,respiratory_rate,systolic_bp,spo2,temperature,label,age"]
    for i in range(16):
        creatinine = 3.0 if i % 4 == 0 else 0.8
        label = 1 if creatinine > 2 else 0
        age = "abc" if i == 0 else "70"       # 首行非数值
        lines.append(f"enc-{i + 1:04d},{creatinine},70,16,120,98,36.8,{label},{age}")
    holdout.write_text("\n".join(lines) + "\n", encoding="utf-8")
    registry = tmp_path / "registry"
    old_argv = sys.argv
    sys.argv = [
        "manage_model.py", "--registry-dir", str(registry),
        "promote", "--artifact", str(artifact),
        "--target-id", "deterioration_24h", "--input", str(holdout),
    ]
    try:
        manage_main()   # 不得因 age 抛裸 ValueError
    finally:
        sys.argv = old_argv
    report = json.loads(
        (registry / "deterioration_24h" / "age-v1" / "eval.json").read_text(encoding="utf-8")
    )
    subgroup = report["rules"]["subgroup"]
    assert subgroup["status"] == "skipped"
    assert "non-numeric" in subgroup["reason"] or "missing" in subgroup["reason"]
```

（第三行 age 用例同时覆盖部分空：把首行 age 改为 `""` 再跑一遍同样断言，reason 同为值缺失措辞、不含 "absent"——可参数化 `"abc"` 与 `""` 两种值。若 promote 成功路径要求 `split` 键等其他参数，以 :399-431 实际 argv 为准补齐。）

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_manage_model.py::test_gate_rejects_input_class_errors_with_unified_message tests/test_manage_model.py::test_gate_age_non_numeric_skips_rule_instead_of_crashing -q`
Expected: FAIL（文件不存在 → 裸 FileNotFoundError traceback；age "abc" → 裸 ValueError）

- [ ] **Step 3: 实现（scripts/manage_model.py）**

1. 读文件（:116-117）：

```python
    try:
        with input_csv.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise SystemExit(f"promotion gate rejected: cannot read holdout CSV: {exc}") from exc
```

2. `evaluate_rows`（:126）：

```python
    try:
        labels, scores = evaluate_rows(rows, engine, "label")
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"promotion gate rejected: {exc}") from exc
```

3. news2 基线计算（:139-143，把字典推导所在表达式包起来）：

```python
            try:
                news2_score({f: float(row[f]) for f in FEATURES if row.get(f, "") not in ("", None)})[0]
            except (TypeError, ValueError, KeyError) as exc:
                raise SystemExit(
                    f"promotion gate rejected: news2 baseline needs numeric vitals: {exc}"
                ) from exc
```

4. age 块（:163-175）整段替换：

```python
    ages = [row.get("age", "") for row in rows]
    numeric_ages: list[float] = []
    age_complete = True
    for age in ages:
        if age in ("", None):
            age_complete = False
            break
        try:
            numeric_ages.append(float(age))
        except (TypeError, ValueError):
            age_complete = False
            break
    if age_complete:
        groups = ["older" if age >= 65 else "younger" for age in numeric_ages]
        diff = subgroup_sensitivity_diff(
            labels, scores, groups, engine.thresholds["high"], min_positives=10
        )
        rules["subgroup"] = (
            {"status": "skipped", "reason": "insufficient positives per group"}
            if diff is None
            else {"status": "pass" if diff <= 0.10 else "fail", "sensitivity_diff": diff}
        )
    elif any(age not in ("", None) for age in ages):
        # 列在但值缺失/非数值——与「整列缺失」措辞区分（P7 deferred 1）。
        rules["subgroup"] = {
            "status": "skipped",
            "reason": "age column present but values missing/non-numeric",
        }
    else:
        rules["subgroup"] = {"status": "skipped", "reason": "age column absent"}
```

- [ ] **Step 4: 跑测试确认通过 + 全套**

Run: `.venv/Scripts/python -m pytest tests/test_manage_model.py tests/test_report_card.py tests/test_production_controls.py -q`，再全套
Expected: 既有 `test_gate_rules_degrade_when_columns_absent`（缺列 → "age column absent"）与 report_card 测试保持绿。

- [ ] **Step 5: Commit**

```bash
git add scripts/manage_model.py tests/test_manage_model.py
git commit -m "fix: unify promotion-gate input rejections and degrade age rule gracefully (P6 deferred 7, P7 deferred 1)"
```

---

### Task 8: make_golden 默认工件路径解耦（P6-8）

**Files:**
- Modify: `scripts/make_golden.py:24-27`
- Test: `tests/test_serving_consistency.py` 或 `tests/test_training.py`

**Interfaces:**
- Produces: `scripts.make_golden.DEFAULT_ARTIFACTS: dict[str, Path]` 值取自 `Settings.model_fields[...].default`（类字段默认，**不用实例**——实例会读环境变量污染 golden 语义）。

- [ ] **Step 1: 写失败测试**

```python
def test_make_golden_default_artifacts_track_settings_defaults():
    """P6 deferred 8：make_golden 的工件路径不得与 app.config 默认值双写。"""
    from app.config import Settings
    from scripts.make_golden import DEFAULT_ARTIFACTS

    assert DEFAULT_ARTIFACTS["deterioration_24h"] == Path(
        Settings.model_fields["model_path"].default
    )
    assert DEFAULT_ARTIFACTS["readmission_30d"] == Path(
        Settings.model_fields["readmission_model_path"].default
    )
```

- [ ] **Step 2: 跑测试确认失败**

Expected: FAIL（断言通过实现细节巧合相等的话即绿——若 Step 1 直接通过，说明双写恰好相等，把断言改为结构性检查：`assert "data/model.json" not in Path("scripts/make_golden.py").read_text(encoding="utf-8")`，钉住"字面量不得回流"，仍算有效红→绿。）

- [ ] **Step 3: 实现**

`scripts/make_golden.py:24-27`：

```python
# 路径单一来源 = app.config 字段默认（P6 deferred 8：改 config 默认值时
# make_golden 不再静默指向旧工件）。用类字段默认而非实例，避免环境变量污染。
DEFAULT_ARTIFACTS = {
    "deterioration_24h": Path(Settings.model_fields["model_path"].default),
    "readmission_30d": Path(Settings.model_fields["readmission_model_path"].default),
}
```

（`from app.config import Settings` 已在 :15 导入。）

- [ ] **Step 4: 跑测试确认通过 + `make_golden --check` + 全套**

Run: `.venv/Scripts/python scripts/make_golden.py --check`

- [ ] **Step 5: Commit**

```bash
git add scripts/make_golden.py tests/<file>
git commit -m "fix: derive golden default artifacts from Settings field defaults (P6 deferred 8)"
```

---

### Task 9: binary_metrics 恒分 AUPRC 平局语义修复 + 门禁 fixture 同步（P6-9）

**Files:**
- Modify: `scripts/evaluate_model.py:47-95`（AP 累积段 :71-82）
- Modify: `tests/test_evaluation.py`（:271-302 `_quadratic_auprc_reference` 同步、新增恒分用例）
- Modify: `tests/test_manage_model.py:6-9`（`_write_gate_rows` docstring）
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `binary_metrics(labels, scores, threshold)` 返回 dict 不变；`auprc` 语义改为 sklearn `average_precision_score` 的平局分组口径：**同分组在组边界统一计 precision×Δrecall；恒分输入 → prevalence**（修复前 1.0）。
- Produces: 门禁/评估/亚组所有 AUPRC 消费方自动获得公平 baseline（恒分 baseline 从虚高 1.0 落到 prevalence）。

**已知影响面（探查确认）**：
- `tests/test_evaluation.py:6-9` 断言 `auprc == 1.0` 的用例分数**无平局** → 修复后仍 1.0，不动。
- `_write_gate_rows` 的模型分数由 creatinine 两点定线且与标签同源 → 模型 AP 仍 1.0；news2 baseline 恒定 → `baseline_auprc` 0.25→ 由 1.0 变 0.25；`invert_labels` 拒绝路径 baseline 也变 0.25 → **必须跑拒绝用例确认仍拒绝**。

- [ ] **Step 1: 写失败测试（红）**

`tests/test_evaluation.py` 新增：

```python
def test_binary_metrics_constant_scores_auprc_is_prevalence():
    """P6 deferred 9：恒分输入的 AUPRC 必须是 prevalence，
    不得经 (score,label) 元组平局排序得 1.0。"""
    from scripts.evaluate_model import binary_metrics

    labels = [1, 0, 1, 0]
    scores = [0.5, 0.5, 0.5, 0.5]
    result = binary_metrics(labels, scores, 0.5)
    assert result["auprc"] == pytest.approx(0.5)


def test_binary_metrics_auprc_matches_tie_aware_reference():
    """平局分组口径对照：实现 == 独立参照 == sklearn（三角钉住）。"""
    from sklearn.metrics import average_precision_score

    from scripts.evaluate_model import binary_metrics
    from tests.test_evaluation import _tie_aware_auprc_reference

    cases = [
        ([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6]),          # 无平局
        ([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]),          # 全平局
        ([1, 0, 1, 0, 1], [0.9, 0.9, 0.7, 0.7, 0.7]),  # 部分平局
        ([1, 1, 0, 0, 0], [0.8, 0.8, 0.8, 0.3, 0.1]),  # 平局组含同类
    ]
    for labels, scores in cases:
        got = binary_metrics(labels, scores, 0.5)["auprc"]
        assert got == pytest.approx(_tie_aware_auprc_reference(labels, scores), abs=1e-12)
        assert got == pytest.approx(average_precision_score(labels, scores), abs=1e-12)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_evaluation.py -q -k "prevalence or tie_aware"`
Expected: FAIL（恒分得 1.0；`_tie_aware_auprc_reference` 尚不存在）

- [ ] **Step 3: 实现**

1. **独立参照**（`tests/test_evaluation.py`，新增在旧 `_quadratic_auprc_reference` 旁；旧函数保留给无平局用例）：

```python
def _tie_aware_auprc_reference(labels: list[int], scores: list[float]) -> float:
    """独立参照：按分数降序分组、组边界统一累积的 average precision
    （与 sklearn average_precision_score 同语义，P6 deferred 9）。"""
    positives = sum(labels)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ap = 0.0
    tp = seen = 0
    recall_prev = 0.0
    i = 0
    while i < len(order):
        score_value = scores[order[i]]
        group_tp = group_n = 0
        while i < len(order) and scores[order[i]] == score_value:
            group_tp += labels[order[i]]
            group_n += 1
            i += 1
        tp += group_tp
        seen += group_n
        recall = tp / positives
        ap += (tp / seen) * (recall - recall_prev)
        recall_prev = recall
    return ap
```

2. **实现**（`scripts/evaluate_model.py:71-82` 替换，:71-73 注释一并改写）：

```python
    # AUPRC（average precision）按分数分组单遍累积 O(n log n)：同分组在组边界统一
    # 计 precision×Δrecall（sklearn 平局语义）。修复前逐行排序在平局时让 label=1
    # 置前、恒分输入得 1.0（P6 deferred 9）；恒分现在得 prevalence。
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    cumulative_tp = 0
    seen = 0
    auprc = 0.0
    previous_recall = 0.0
    index = 0
    while index < len(ordered):
        score_value = ordered[index][0]
        group_tp = 0
        group_size = 0
        while index < len(ordered) and ordered[index][0] == score_value:
            group_tp += ordered[index][1]
            group_size += 1
            index += 1
        cumulative_tp += group_tp
        seen += group_size
        precision = cumulative_tp / seen
        recall = cumulative_tp / positives if positives else 0.0
        auprc += precision * max(recall - previous_recall, 0)
        previous_recall = recall
```

3. `tests/test_evaluation.py:271-302`：把 `_quadratic_auprc_reference` 参与**有平局数据**的对照断言改为 `_tie_aware_auprc_reference`（无平局对照保留——两参照在无平局时数值相同）。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python -m pytest tests/test_evaluation.py -q`
Expected: 全绿（含既有 100k 行 `0 < auprc <= 1` 用例与位级对照用例）

- [ ] **Step 5: 门禁 fixture 同步 + 应急步骤**

1. 更新 `tests/test_manage_model.py:6-9` docstring：

```python
    """16 行留出集、25% 阳性：空阳性 bootstrap 抽样 0.75^16≈1.0% < 2.5% 分位，
    保证 CI lo 可用；news2 分量恒定 → 平局分组口径下 baseline_auprc = prevalence
    0.25（P6 deferred 9 修复后；修复前元组平局虚高为 1.0）。
    模型分数由 creatinine 两点定线且与标签同源 → 模型 AUPRC 恒为 1.0。"""
```

2. Run: `.venv/Scripts/python -m pytest tests/test_manage_model.py tests/test_production_controls.py -q`
   - 预期全绿（模型 AP 仍 1.0 ≥ 0.25；拒绝路径模型 AP < 0.25 仍拒绝）。
   - **应急**：若 `invert_labels` 拒绝用例失效（模型 AP 的 CI lo 意外 ≥ baseline），在 `_write_gate_rows` 把 invert 情形的标签翻转扩展到两档 creatinine 使模型排序与标签完全反向（AP→近 0），断言语义不变。
3. 同步检查 `evaluate`/`report_card` 相关断言：Run `pytest tests/test_evaluation.py tests/test_report_card.py tests/test_validation.py -q`。

- [ ] **Step 6: 全套回归 + ruff**

Run: `.venv/Scripts/python -m pytest -q`（含 @slow 冒烟——门禁 baseline 变化会流经 1500 规模演练，预期 pass：baseline 变低只会让"模型≥基线"更易满足）
Expected: 全绿。若冒烟 gate 用例断言了 baseline 具体数值 → 按新语义更新其期望值（prevalence 口径），并在提交说明里列出改动的数值断言。

- [ ] **Step 7: Commit**

```bash
git add scripts/evaluate_model.py tests/test_evaluation.py tests/test_manage_model.py
git commit -m "fix: tie-aware average precision — constant scores yield prevalence, not 1.0 (P6 deferred 9)"
```

---

### Task 10: Optuna subsample 配对 subsample_freq（P7-2）

**Files:**
- Modify: `scripts/train_model.py:394-403`（objective 内 params → 抽出 `_lgbm_search_space`）、:437 附近（`study.best_params` → 最终模型）
- Test: `tests/test_lightgbm.py`

**Interfaces:**
- Produces: `_lgbm_search_space(trial) -> dict` —— 搜索空间构建函数（不 import optuna，trial 走 duck typing）；空间恒含 `subsample_freq ≥ 1`，最终模型参数 = `{**study.best_params, ...}` 含生效的 `subsample_freq`。

- [ ] **Step 1: 写失败测试**

`tests/test_lightgbm.py` 新增：

```python
def test_lgbm_search_space_pairs_subsample_with_freq():
    """P7 deferred 2：subsample 必须配 subsample_freq>0，否则 LightGBM 忽略 bagging，
    整个搜索维度无效。"""

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

    space = _lgbm_search_space(_StubTrial())
    assert "subsample" in space and "subsample_freq" in space
    assert space["subsample_freq"] >= 1  # LightGBM 默认 0 = 不启用 bagging
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_lightgbm.py::test_lgbm_search_space_pairs_subsample_with_freq -q`
Expected: FAIL（ImportError）

- [ ] **Step 3: 实现**

`scripts/train_model.py`，在 `_fit_lightgbm` 外新增：

```python
def _lgbm_search_space(trial) -> dict:
    """Optuna 搜索空间（P7 deferred 2：subsample 必须配 subsample_freq≥1，
    否则 LightGBM 默认 freq=0 直接忽略 bagging，该维度搜索无效）。"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
```

objective 内（:394-403）替换为 `params = _lgbm_search_space(trial)`。

**必须核对**：读 :430-445，`study.best_params` 传给最终 `LGBMClassifier` 处——`suggest_int` 的 `subsample_freq` 会自然进入 `best_params` ✓；若最终模型用**独立的 params dict** 而非 `best_params`，则在该 dict 中显式补 `"subsample_freq": 1`（原缺陷即在此）。

- [ ] **Step 4: 跑测试确认通过 + 全套**

Run: `.venv/Scripts/python -m pytest tests/test_lightgbm.py -q`，再全套
Expected: 全绿（Optuna 路径无既有单测，`@slow` 冒烟不含 lightgbm；若本机装了 optuna 可手动 `--algorithm lightgbm --trials 2` 冒烟——非必需）。

- [ ] **Step 5: Commit**

```bash
git add scripts/train_model.py tests/test_lightgbm.py
git commit -m "fix: pair Optuna subsample with subsample_freq so bagging is actually active (P7 deferred 2)"
```

---

### Task 11: per_facility_auprc 全跳过 → unchecked；亚组差取全体 max-min（P7-3）

**Files:**
- Modify: `scripts/validate.py:171-195`（subgroup 尾部）、:198-229（facility）
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: 既有签名 `subgroup_sensitivity_diff(labels, scores, groups, threshold, *, min_positives=10) -> float | None`、`per_facility_auprc(rows, labels, scores, baseline_auprc, *, facility_column="facility_id", min_rows=20, min_positives=5) -> dict`。
- Produces: 全部院区被跳过时返回 `{"checked": False, "worst": None, "failures": [], "reason": "all facilities skipped (min_rows/min_positives)"}`——消费方（manage_model:177-179、report_card:377-378）既有 `checked=False` 分支走 skipped；`subgroup_sensitivity_diff` ≥3 组返回 `max(灵敏度) - min(灵敏度)`。

- [ ] **Step 1: 写失败测试（2 个）**

`tests/test_validation.py` 新增：

```python
def test_per_facility_auprc_all_skipped_is_unchecked():
    """P7 deferred 3a：全部院区因 min_rows/min_positives 被跳过 → checked=False，
    不得当成 pass（worst=None 会让消费方误判通过）。"""
    from scripts.validate import per_facility_auprc

    rows = [{"facility_id": "A"}] * 5 + [{"facility_id": "B"}] * 5
    labels = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    scores = [0.9, 0.1, 0.2, 0.3, 0.4, 0.8, 0.1, 0.2, 0.3, 0.4]
    result = per_facility_auprc(rows, labels, scores, 0.5, min_rows=20, min_positives=5)
    assert result["checked"] is False
    assert "skipped" in result["reason"]
    assert result["failures"] == []


def test_subgroup_sensitivity_diff_considers_all_groups():
    """P7 deferred 3b：≥3 组取全体 max-min——前两组相同、第三组失衡必须被抓出。"""
    from scripts.validate import subgroup_sensitivity_diff

    groups = ["g1"] * 20 + ["g2"] * 20 + ["g3"] * 20
    labels = [1] * 10 + [0] * 10 + [1] * 10 + [0] * 10 + [1] * 10 + [0] * 10
    # g1/g2 灵敏度 1.0，g3 灵敏度 0.0
    scores = [0.9] * 10 + [0.1] * 10 + [0.9] * 10 + [0.1] * 10 + [0.1] * 10 + [0.9] * 10
    diff = subgroup_sensitivity_diff(labels, scores, groups, 0.5, min_positives=10)
    assert diff == pytest.approx(1.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_validation.py -q -k "all_skipped or all_groups"`
Expected: FAIL（现状返回 `checked=True`；三组 diff 返回 0.0）

- [ ] **Step 3: 实现（scripts/validate.py）**

1. `subgroup_sensitivity_diff` 末行（:195）：

```python
    # P7 deferred 3b：≥3 组时取全体 max-min，不再只比较前两组。
    return max(sensitivities) - min(sensitivities)
```

2. `per_facility_auprc`（:217-229）加计数：

```python
    failures = []
    worst = None
    evaluated = 0
    for facility, indices in sorted(buckets.items()):
        if len(indices) < min_rows:
            continue
        facility_labels = [labels[i] for i in indices]
        if sum(facility_labels) < min_positives:
            continue
        evaluated += 1
        auprc = binary_metrics(facility_labels, [scores[i] for i in indices], 0.5)["auprc"] or 0.0
        worst = auprc if worst is None else min(worst, auprc)
        if auprc < baseline_auprc:
            failures.append({"facility": facility, "auprc": auprc, "baseline": baseline_auprc})
    if not evaluated:
        # P7 deferred 3a：全部院区被跳过 → 未评估，不得报 checked=True/pass。
        return {
            "checked": False,
            "worst": None,
            "failures": [],
            "reason": "all facilities skipped (min_rows/min_positives)",
        }
    return {"checked": True, "worst": worst, "failures": failures}
```

- [ ] **Step 4: 跑测试确认通过 + 全套**

Run: `.venv/Scripts/python -m pytest tests/test_validation.py tests/test_manage_model.py tests/test_report_card.py -q`，再全套
Expected: 既有两院区 pass/fail 用例保持绿（它们 `evaluated ≥ 1`）。

- [ ] **Step 5: Commit**

```bash
git add scripts/validate.py tests/test_validation.py
git commit -m "fix: facility rule unchecked when all skipped; subgroup diff spans all groups (P7 deferred 3)"
```

---

### Task 12: recalibrate 标签严格 0/1 + gate-input 预检（P7-5、P7-8）

**Files:**
- Modify: `scripts/recalibrate.py:63-72`（抽出 `_validated_labels`）、:95-97（gate-input 预检）
- Test: `tests/test_drift.py`

**Interfaces:**
- Consumes: 既有 `run_promotion_gate(candidate_path, target_id, gate_path)`（门禁内部检查保留为其他调用方的兜底）。
- Produces: `_validated_labels(rows, *, context: str) -> list[int]` —— 严格接受字符串 `"0"`/`"1"`（`int("2")`、`"0.5"`、`"-1"`、`"1.0"` 全拒，拒绝语 `f"recalibrate: row {i} needs label 0/1"`）；单类拒绝语 `f"recalibrate: {context} needs both classes"`（context ∈ {"--input", "gate-input"}）。`--gate-input` 在进门前禁前预检。

- [ ] **Step 1: 写失败测试（红）**

`tests/test_drift.py` 新增：

```python
def test_recalibrate_rejects_non_binary_labels():
    """P7 deferred 8：int("2")==2 曾被放行进校准拟合——严格只收 "0"/"1"。"""
    import pytest
    # 复用 test_recalibrate_only_updates_calibration_and_gates 的工件/CSV 构造，
    # 仅把 label 列改为参数化值：
    for bad in ("2", "0.5", "-1", "1.0"):
        rows_csv = <同该测试的输入，首行 label 置为 bad，其余合法>
        <invoke recalibrate main（argv 模式同既有测试）>
        # 用 pytest.raises 包裹——见下


def test_recalibrate_gate_input_prechecked_with_recalibrate_message(tmp_path):
    """P7 deferred 5：全零 gate-input 应在 recalibrate 输入校验处即拒，
    拒绝语出自 recalibrate 而非门禁度量入口。"""
    # 全零 --gate-input → SystemExit match "recalibrate: gate-input needs both classes"
    # 原件 calibration 块分毫不动（沿用既有测试的原件比对断言）
```

Step 1 落地要求：参数化拒绝用例写成：

```python
@pytest.mark.parametrize("bad_label", ["2", "0.5", "-1", "1.0"])
def test_recalibrate_rejects_non_binary_labels(tmp_path, bad_label):
    import pytest
    # 构造最小 labeled CSV：复用 tests/test_drift.py 既有 recalibrate 测试的
    # 工件与行模板，仅首行 label = bad_label
    ...
    with pytest.raises(SystemExit, match="needs label 0/1"):
        <recalibrate 调用>
```

（具体工件/argv 复制自 `tests/test_drift.py:71-169` 的既有测试，改 label 值。）

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_drift.py -q -k "non_binary or prechecked"`
Expected: FAIL（"2"/"1.0" 被放行；全零 gate 的拒绝语是门禁的 "holdout needs..." 不匹配新 match）

- [ ] **Step 3: 实现（scripts/recalibrate.py）**

1. 模块级新增：

```python
def _validated_labels(rows: list[dict], *, context: str) -> list[int]:
    """严格 0/1 校验（P7 deferred 8：int("2")==2 曾被放行）。
    context 进拒绝语，区分 --input 与 gate-input（P7 deferred 5）。"""
    labels: list[int] = []
    for index, row in enumerate(rows):
        raw = row.get("label")
        text = raw.strip() if isinstance(raw, str) else raw
        if text not in ("0", "1"):
            raise SystemExit(f"recalibrate: row {index + 1} needs label 0/1")
        labels.append(int(text))
    if not labels or not sum(labels) or sum(labels) == len(labels):
        raise SystemExit(f"recalibrate: {context} needs both classes")
    return labels
```

2. main 内 :65-72 替换为 `labels = _validated_labels(rows, context="--input")`。
3. gate-input 预检（在调 `run_promotion_gate` 之前、engine 校验之后即可）：

```python
    gate_path = Path(args.gate_input or args.input)
    if args.gate_input:
        try:
            with gate_path.open(newline="", encoding="utf-8-sig") as handle:
                gate_rows = list(csv.DictReader(handle))
        except OSError as exc:
            raise SystemExit(f"recalibrate: cannot read gate-input: {exc}") from exc
        _validated_labels(gate_rows, context="gate-input")
```

并把 :96 的 `Path(args.gate_input or args.input)` 换成 `gate_path`。

4. 门禁内部 `manage_model.py:126-128` 的既有检查**保留**（其他调用方兜底）。

- [ ] **Step 4: 同步既有测试期望（红变绿的合法更新）**

`tests/test_drift.py:140-169`（全零 gate-input 用例）：若其 `pytest.raises` 带 `match` 指向门禁旧文案，改为：

```python
        with pytest.raises(SystemExit, match="recalibrate: gate-input needs both classes"):
```

若原是裸 `raises(SystemExit)` 则不动（新拒绝同样命中）。docstring 中「拒绝语来自门禁」的注释同步改为「来自 recalibrate 预检」。

- [ ] **Step 5: 跑测试确认通过 + 全套**

Run: `.venv/Scripts/python -m pytest tests/test_drift.py -q`，再全套
Expected: 全绿（校准拟合路径不再收到非 0/1 标签）。

- [ ] **Step 6: Commit**

```bash
git add scripts/recalibrate.py tests/test_drift.py
git commit -m "fix: strict 0/1 label parsing and gate-input precheck in recalibrate (P7 deferred 5/8)"
```

---

### Task 13: readmission isotonic Brier 不劣化断言（P7-6）

**Files:**
- Modify: `tests/test_training.py:204-217`（`test_readmission_uses_isotonic_calibration` 扩展）

**Interfaces:**
- Consumes: 既有 `_train(tmp_path, extra=["--target-id", "readmission_30d"])`、`build_demo_dataset`、`binary_metrics`、`RiskEngine`；参照 `test_calibration_improves_or_ties_brier`（:88-146）的「双引擎对比」写法。

**关键保证**（执行者须知）：isotonic 在其拟合样本上单调最小化平方误差，而未校准分数是 margin 的单调函数 → 只要评估数据 == 拟合数据（同 seed 重建），`brier_cal <= brier_raw` 数学上成立、不会 flaky。**必须用 `build_demo_dataset("readmission_30d", 480)` 重建**——`_train` 固定 `--rows 480`（tests/test_training.py:16）。

- [ ] **Step 1: 写失败测试（红）**

`tests/test_training.py`，在 `test_readmission_uses_isotonic_calibration` 末尾（既有形状断言之后）追加：

```python
    # P7 deferred 6：真 isotonic 分支此前从未被 Brier 不劣化钉住（deterioration
    # 走的是 R-P7-5a Platt 回退）。评估数据与拟合数据同源（同 seed 同行数重建）。
    import copy

    from app.domain import ClinicalEvent
    from app.risk import RiskEngine
    from scripts.evaluate_model import binary_metrics
    from scripts.train_model import build_demo_dataset

    payloads, _matrix, raw_labels = build_demo_dataset(
        "readmission_30d", 480  # 必须与 _train 的 --rows 480 一致（tests/test_training.py:16）
    )
    calibrated_engine = RiskEngine(
        Settings(readmission_model_path=path), target_id="readmission_30d"
    )
    raw_artifact = copy.deepcopy(artifact)
    raw_artifact["calibration"] = None
    raw_path = tmp_path / "raw_readmission.json"
    raw_path.write_text(json.dumps(raw_artifact), encoding="utf-8")
    raw_engine = RiskEngine(
        Settings(readmission_model_path=str(raw_path)), target_id="readmission_30d"
    )

    cal_scores, raw_scores = [], []
    for i, payload in enumerate(payloads):
        event = ClinicalEvent(
            event_type="discharge",
            patient_id=f"iso-p-{i}",
            encounter_id=f"iso-e-{i}",
            payload=payload,
        )
        cal_scores.append(calibrated_engine.score(event).score)
        raw_scores.append(raw_engine.score(event).score)
    brier_cal = binary_metrics(raw_labels, cal_scores, calibrated_engine.thresholds["high"])["brier"]
    brier_raw = binary_metrics(raw_labels, raw_scores, raw_engine.thresholds["high"])["brier"]
    assert brier_cal <= brier_raw + 1e-9
```

（`Settings` 若该测试文件尚未 import，在函数内 `from app.config import Settings`；`build_demo_dataset` 对 readmission 返回 `(times, matrix, labels)`，`times` 即 payload 字典序列——参考 `scripts/make_golden.py:44-49` 的用法。）

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_training.py::test_readmission_uses_isotonic_calibration -q`
Expected: FAIL（当前无 Brier 断言 → ImportError/AttributeError 之外的失败形态应为断言缺失；若追加即绿，说明数据没对上拟合集——核对 `_train` 的 size 常量与 `build_demo_dataset` 调用参数完全一致后再跑。）

- [ ] **Step 3: 实现**

本任务为纯测试增强（追加上述断言）；实现无需改动。若 Step 2 显示 isotonic 真的劣化 Brier（数据同源前提下不应发生）→ 停下报告，不得弱化断言。

- [ ] **Step 4: 跑测试确认通过 + 全套**

- [ ] **Step 5: Commit**

```bash
git add tests/test_training.py
git commit -m "test: pin readmission isotonic calibration Brier non-regression (P7 deferred 6)"
```

---

### Task 14: architecture.md 补漂移输入 schema 与 R-P7-5a 校准回退两节（P7-4、P7-7）

**Files:**
- Modify: `docs/architecture.md`（:34-39 漂移监控节；新增校准小节）

**Interfaces:** 纯文档，无代码消费方。文档事实依据：`scripts/monitor_drift.py:67-79,132-140`、`app/features.py:103-118`、`scripts/train_model.py:442-470`。

- [ ] **Step 1: 写文档（失败测试不适用——本任务为 spec 型修复，用事实核查清单代替红灯）**

在 `docs/architecture.md` 批量漂移检查段落（:37 附近）之后补充：

```markdown
**漂移输入 CSV 的 schema 要求**（`monitor_drift`）：输入须包含工件 `feature_order` 的全部特征列——deterioration 为 21 维 v2（10 原始体征 + 10 个 `__missing` 指示列 + `news2_score` 派生列）；readmission 为 6 维出院快照编码。readmission 的派生列（`diagnosis_code_count`、`discharge_destination_code`）缺失时按出院源字段（`discharge_destination`、`los_days`、`diagnosis_codes`、`comorbidity_index`、`prior_utilization_12m`、`ed_visits_6m`）即时编码（R-RS-15）；其他缺失列按缺失入 bucket 计入 missing rate，不报错。
```

在文档适当位置（再校准/模型训练章节）新增：

```markdown
**校准方法与分辨率回退（Ruling R-P7-5/5a）**：校准默认 isotonic 回归；当拟合段唯一 y 水平 < 3（分数会坍缩成二值、摧毁阈值阶梯与分带）时自动回退 Platt，并在工件 `calibration` 块记录 `fallback_reason: isotonic_resolution`。demo deterioration 触发该回退属预期；真实数据阳性数充足时走真 isotonic。
```

- [ ] **Step 2: 事实核查（对照实现逐句确认）**

对照 `scripts/monitor_drift.py`（列名/派生编码/缺失行为）、`app/features.py`（两派生列名与源字段清单）、`scripts/train_model.py:442-470`（回退条件与 `fallback_reason` 值）逐句核对文档新增文字——任何对不上就改文档（不改代码）。

- [ ] **Step 3: Commit**

```bash
git add docs/architecture.md
git commit -m "docs: drift input CSV schema and isotonic resolution fallback (P7 deferred 4/7)"
```

---

### Task 15: 删除死常量 READMISSION_POSITIVE_RATE（P8-1）

**Files:**
- Modify: `scripts/train_model.py:44`
- Test: `tests/test_training.py`

- [ ] **Step 1: 写失败测试（红）**

```python
def test_readmission_positive_rate_dead_constant_removed():
    """P8 deferred 1：P6 均匀 8% 抽样残留，P8 字段驱动标签后成死常量——删除并钉住。"""
    import scripts.train_model as train_model

    assert not hasattr(train_model, "READMISSION_POSITIVE_RATE")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_training.py::test_readmission_positive_rate_dead_constant_removed -q`
Expected: FAIL（常量仍存在）

- [ ] **Step 3: 实现**

删除 `scripts/train_model.py:44` 的 `READMISSION_POSITIVE_RATE = 0.08` 一行（先全仓 grep 确认 scripts/app/tests 零引用——docs/STATUS/旧计划中的提及是历史记录不删）。

- [ ] **Step 4: 跑测试确认通过 + 全套 + ruff**

- [ ] **Step 5: Commit**

```bash
git add scripts/train_model.py tests/test_training.py
git commit -m "chore: remove dead READMISSION_POSITIVE_RATE constant (P8 deferred 1)"
```

---

### Task 16: import summary 补 discharge_events（P8-2）

**Files:**
- Modify: `docs/data-integration-design.md:159`
- Test: `tests/test_import.py`

**Interfaces:**
- Consumes: 既有 `scripts/import_clinical_data.py:70-76` summary（键：`patients`、`events`、`discharge_events`、`commit`、`event_types`；`--commit` 后追加 `accepted_events`、`duplicate_events`）。

- [ ] **Step 1: 写失败测试（红）**

`tests/test_import.py` 新增（构造沿用 :12-39 的最小 patients/events，事件含 1 条 discharge）：

```python
def test_import_summary_counts_discharge_events(tmp_path):
    """P8 deferred 2：summary 实际含 discharge_events 键——文档与实现对齐。"""
    # 跑 dry-run 导入（同既有测试的调用方式），解析 stdout JSON summary
    summary = <既有 dry-run 测试的 summary 解析方式>
    assert summary["discharge_events"] == 1
    assert {"patients", "events", "discharge_events", "commit", "event_types"} <= set(summary)
```

（具体调用复用 `tests/test_import.py` 中已有 dry-run 用例的构造。）

- [ ] **Step 2: 跑测试确认失败**

Expected: FAIL 于**文档断言不可测的部分不适用**——测试本身对实现应直接为绿（实现已有该键）。**本项的红灯** = 文档 diff：先做 Step 3 改文档，再把测试跑绿一起提交；测试是防实现回退的钉子。若实现真缺该键 → 先修实现。

- [ ] **Step 3: 改文档**

`docs/data-integration-design.md:159` 字段清单：

```markdown
summary 输出字段：`patients`、`events`、`discharge_events`、`commit`、`event_types`；`--commit` 后追加 `accepted_events`、`duplicate_events`（T2.4 落地后中途失败会附 partial summary：accepted/failed 计数 + 最后成功键）。
```

- [ ] **Step 4: 跑测试确认通过 + 全套**

- [ ] **Step 5: Commit**

```bash
git add docs/data-integration-design.md tests/test_import.py
git commit -m "docs: add discharge_events to import summary field list, pin with test (P8 deferred 2)"
```

---

### Task 17: promote 白名单 readmission 旧形态回归测试（P8-3）

**Files:**
- Test: `tests/test_manage_model.py`

**Interfaces:**
- Consumes: `scripts.manage_model.load_artifact(path, target_id, version=None)`（:51-68）；白名单 readmission 半边 = `{tuple(FEATURES), tuple(READMISSION_FEATURES)}`（`FEATURES` 来自 `app.synthetic`，10 维旧 vitals）。

- [ ] **Step 1: 写失败测试（红——缺测试即失败的「覆盖钉」，用新测试文件段落落地）**

`tests/test_manage_model.py` 新增：

```python
def test_load_artifact_accepts_legacy_vitals_readmission_artifact(tmp_path):
    """P8 deferred 3：readmission 白名单半边（旧 vitals 形态）必须有回归测试。"""
    import pytest

    from app.synthetic import FEATURES
    from scripts.manage_model import load_artifact

    legacy = tmp_path / "legacy_readmission.json"
    legacy.write_text(
        json.dumps({"target_id": "readmission_30d", "feature_names": list(FEATURES)}),
        encoding="utf-8",
    )
    artifact = load_artifact(legacy, "readmission_30d")
    assert tuple(artifact["feature_names"]) == tuple(FEATURES)


def test_load_artifact_rejects_unknown_signature(tmp_path):
    """负例：既非旧 vitals 也非出院快照编码的签名必须响亮拒绝。"""
    import pytest

    from scripts.manage_model import load_artifact

    bad = tmp_path / "bad_sig.json"
    bad.write_text(
        json.dumps({"target_id": "readmission_30d", "feature_names": ["bogus_feature"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model feature signature is invalid"):
        load_artifact(bad, "readmission_30d")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_manage_model.py -q -k "legacy_vitals or unknown_signature"`
Expected: 若 `load_artifact` 要求更多字段（如 `weights`）而 KeyError → 按实际校验路径补最小字段（保持签名白名单断言语义）；新测试缺位为本项缺陷本身——测试落地后两用例应直接证明白名单双半边（先跑确认现有实现通过或暴露真实缺口，真实缺口 → 报告并修实现）。

- [ ] **Step 3: 实现**

本任务默认纯测试（白名单实现已在 R-P8-9 落地）。若 Step 2 暴露真实拒绝 → 修 `load_artifact` 白名单并在提交说明记录。

- [ ] **Step 4: 跑测试确认通过 + 全套**

- [ ] **Step 5: Commit**

```bash
git add tests/test_manage_model.py
git commit -m "test: pin both promote-whitelist halves for readmission artifacts (P8 deferred 3)"
```

---

### Task 18: live 路径 readmission 钉首个出院事件 + 契约对齐（P8-4）

**Files:**
- Modify: `app/worker.py:146-160`（`score_event`）；必要时 `app/api.py:27-33、:283-294`（若存在独立打分循环）
- Modify: `docs/data-contract.md:57`
- Test: `tests/test_worker.py`、`tests/test_api.py`

**Interfaces:**
- Consumes: 既有 `store.events_for(encounter_id, tenant_id, facility_id)`（SQLite/Postgres 双实现都有，按 `occurred_at` 升序）、`targets_for_event(event)`、`upsert_score`（防倒退：新 as_of < 旧 as_of → `skipped_stale`）。
- Produces: live 与批路径（`_first_discharge`，worker:36-40）语义一致——**仅当来的是（已入库）首个出院事件才打 readmission 分**，后续出院事件不重打；deterioration 触发逻辑不变。

**语义边界（契约措辞不得越界）**：承诺顺序到达下的"钉首个出院"；乱序到达（后出院先入库）受 `upsert_score` 防倒退保护，不承诺回改。

- [ ] **Step 1: 审计 api.py**

Grep `app/api.py` 中 readmission 打分路径：若 :27-33 `score_all_targets` 与 :283-294 同步入库后打分**委托** `worker.score_event` → 只改 worker；若有**内联打分循环** → 把下述守卫抽成共享函数（放 `app/worker.py` 导出，api import）。在提交说明记录结论。

- [ ] **Step 2: 写失败测试（红）**

`tests/test_worker.py` 新增（构造沿用 :165-200 既有双事件用例的 store/engines 模板）：

```python
def test_second_discharge_event_does_not_rescore_readmission(store, engines):
    """P8 deferred 4：live 路径与批路径/契约一致——readmission 单次评分、
    as_of 钉首个出院事件；第二个出院事件不重打分。"""
    d1 = ClinicalEvent(
        event_type="discharge", patient_id="p1", encounter_id="e1",
        occurred_at=datetime(2026, 1, 1, 10, 0), payload={...},  # 同既有用例的出院 payload
    )
    d2 = d1.model_copy(
        update={
            "event_id": "evt-d2",
            "occurred_at": datetime(2026, 1, 3, 10, 0),
            "payload": {**<payload>, "los_days": 5},
        }
    )
    store.add_event(d1)
    score_event(store, engines, d1)
    first = store.get_score("e1", d1.tenant_id, d1.facility_id, target_id="readmission_30d")
    assert first is not None and first.as_of == d1.occurred_at

    store.add_event(d2)
    score_event(store, engines, d2)   # 第二个出院 → 不得改写 readmission
    after = store.get_score("e1", d2.tenant_id, d2.facility_id, target_id="readmission_30d")
    assert after.as_of == d1.occurred_at
    assert after.scored_at == first.scored_at   # 未被重打分
    # deterioration 仍随事件正常重打（若该用例 engines 含 deterioration 则断言更新）
```

（`store`/`engines` fixture 或工厂按既有用例 :123-200 的获取方式；`add_event` 以实际 store API 名为准——既有用例都是"先入库再 score_event"，照抄其入库调用。）同步在 `tests/test_api.py` 同步 ingest 路径加同款双出院断言（沿 :527-570 模板）。

- [ ] **Step 3: 跑测试确认失败**

Run: `.venv/Scripts/python -m pytest tests/test_worker.py -q -k "second_discharge"`
Expected: FAIL（现状 `after.as_of == d2.occurred_at`、`scored_at` 变化）

- [ ] **Step 4: 实现（app/worker.py）**

`score_event` 内（:155-159 的 target 循环）：

```python
    for target_id in targets_for_event(event):
        target_engine = engines.get(target_id)
        if target_engine is None:
            continue
        if target_id == "readmission_30d":
            # 契约：readmission 单次评分、as_of 钉首个（按时间）出院事件；
            # 与批路径 _first_discharge 对齐（P8 deferred 4）——后续出院事件不重打分。
            discharges = [
                item
                for item in store.events_for(
                    event.encounter_id, event.tenant_id, event.facility_id
                )
                if item.event_type == "discharge"
            ]
            first = min(discharges, key=lambda item: (item.occurred_at, item.event_id))
            if (first.occurred_at, first.event_id) != (event.occurred_at, event.event_id):
                continue
        store.upsert_score(target_engine.score(scoring_event))
```

（`score_event` 为逐事件函数，`targets_for_event` 已保证走到 readmission 时 `event` 必是出院事件 → `discharges` 非空。若 Step 1 发现 api 有内联循环，抽出 `readmission_should_score(store, event) -> bool` 共享。）

- [ ] **Step 5: 跑测试确认通过 + 契约措辞**

1. Run: `.venv/Scripts/python -m pytest tests/test_worker.py tests/test_api.py -q`
2. `docs/data-contract.md:57` 改为：

```markdown
`readmission_30d` 只在 `event_type == "discharge"` 的出院事件上评分（**单次评分，`as_of` 钉在首个出院事件时刻**，后续生命体征不刷新）；同一就诊的**后续出院事件不重打分**——live 事件路径与批量 reconcile 的口径一致（事件按到达序处理，乱序到达的边界由评分防倒退机制兜底）。非出院事件的 `scores` 只含 deterioration。
```

- [ ] **Step 6: 全套回归**

Expected: 既有单出院用例（test_api:527-570、test_worker:165-247）全绿——它们首出院即唯一出院，守卫条件为真、行为不变。

- [ ] **Step 7: Commit**

```bash
git add app/worker.py app/api.py docs/data-contract.md tests/test_worker.py tests/test_api.py
git commit -m "fix: live readmission scoring pinned to first discharge, aligned with batch path and contract (P8 deferred 4)"
```

（若 Step 1 结论为 api 已委托 worker，则 `app/api.py` 不入提交。）

---

### Task 19: 出院字段双常量交叉相等测试 + 消除手写清单（P8-5）

**Files:**
- Test: `tests/test_features.py`（新增交叉测试）
- Modify: `tests/test_import.py:128-152`（缺失字段断言改为遍历常量）

**Interfaces:**
- Consumes: `app.domain.DISCHARGE_SNAPSHOT_FIELDS`（6 项）、`app.features.READMISSION_PAYLOAD_KEYS`（6 项，集合相等、顺序不同）。

- [ ] **Step 1: 写失败测试（红——测试缺位即缺陷）**

`tests/test_features.py` 新增：

```python
def test_discharge_snapshot_fields_match_readmission_payload_keys():
    """P8 deferred 5：两处 6 字段契约必须集合相等——双常量漂移由本测试拦截。"""
    from app.domain import DISCHARGE_SNAPSHOT_FIELDS
    from app.features import READMISSION_PAYLOAD_KEYS

    assert len(DISCHARGE_SNAPSHOT_FIELDS) == 6
    assert set(DISCHARGE_SNAPSHOT_FIELDS) == set(READMISSION_PAYLOAD_KEYS)
```

- [ ] **Step 2: 重构 `tests/test_import.py:128-152`**

把逐个手写字段名的缺失断言改为遍历（消除第三份手写清单）：

```python
    from app.domain import DISCHARGE_SNAPSHOT_FIELDS

    for field in DISCHARGE_SNAPSHOT_FIELDS:
        payload = dict(full_valid_payload)   # 既有完整合法 payload
        payload.pop(field)
        ...  # 既有「缺失字段被拒」断言，match 用 field
        assert field in str(exc.value)
```

（`full_valid_payload` = 既有 :154-207 合法 payload 的变量；保留其原用例。）

- [ ] **Step 3: 跑测试确认通过**

Run: `.venv/Scripts/python -m pytest tests/test_features.py tests/test_import.py -q`
Expected: 全绿（实现两常量本就相等——本任务为覆盖钉 + 消除漂移风险面；若不相等则为真实缺陷，报告并修实现/文档）。

- [ ] **Step 4: 全套 + Commit**

```bash
git add tests/test_features.py tests/test_import.py
git commit -m "test: pin discharge snapshot and readmission payload field parity (P8 deferred 5)"
```

---

### Task 20: 门禁 LACE 字段缺失 SystemExit 集成测试（P8-6）

**Files:**
- Test: `tests/test_manage_model.py`

**Interfaces:**
- Consumes: 既有 `_write_lace_artifact`（:174-199）、`_write_lace_rows`（:160-172）、promote argv 模式（:399-431）；拒绝链 `baselines.lace_score` ValueError（"lace fields missing/invalid: <field>"）→ `manage_model` 包装 `SystemExit("promotion gate rejected: ...")`（:144-153）。

- [ ] **Step 1: 写失败测试（红——集成覆盖缺位即缺陷）**

`tests/test_manage_model.py` 新增：

```python
def test_promote_gate_rejects_holdout_missing_lace_field(tmp_path):
    """P8 deferred 6：holdout 缺 LACE 必需字段 → SystemExit 统一拒绝语，
    不留 manifest（单元级 lace_score 测试已有，缺 promote→SystemExit 集成链）。"""
    import pytest

    from scripts.manage_model import main as manage_main

    artifact = tmp_path / "readmission.json"
    _write_lace_artifact(artifact, "lace-v1")   # 既有 readmission 工件构造
    holdout = tmp_path / "holdout.csv"
    # 复用既有 _write_lace_rows 的行模板，但删掉 ed_visits_6m 列
    lines = [<同 _write_lace_rows 的 header，去掉 ed_visits_6m>]
    ...  # 数据行同步去列（label/分组键保留）
    holdout.write_text("\n".join(lines) + "\n", encoding="utf-8")
    registry = tmp_path / "registry"
    old_argv = sys.argv
    sys.argv = [
        "manage_model.py", "--registry-dir", str(registry),
        "promote", "--artifact", str(artifact),
        "--target-id", "readmission_30d", "--input", str(holdout),
    ]
    try:
        with pytest.raises(
            SystemExit,
            match="promotion gate rejected: lace fields missing/invalid: ed_visits_6m",
        ):
            manage_main()
    finally:
        sys.argv = old_argv
    assert not (registry / "manifest.json").exists()
```

（实现链已存在（unit 级 test_lace.py:68-81 + manage_model:149-153 包装）→ 测试落地预期直接绿；若红说明集成链有真缺口 → 修 `run_promotion_gate` 的包装覆盖。）

- [ ] **Step 2: 跑测试**

Run: `.venv/Scripts/python -m pytest tests/test_manage_model.py::test_promote_gate_rejects_holdout_missing_lace_field -q`
Expected: 绿（覆盖钉）或红（真缺口 → 修实现，提交说明记录）。

- [ ] **Step 3: 全套 + Commit**

```bash
git add tests/test_manage_model.py
git commit -m "test: integration coverage for gate SystemExit on missing LACE fields (P8 deferred 6)"
```

---

### Task 21: 收尾 — 状态归档与终验

**Files:**
- Modify: `docs/superpowers/STATUS.md`（Deferred 节、验证基线数字 194→196）

- [ ] **Step 1: 全量终验**

```bash
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m ruff check .
.venv/Scripts/python scripts/make_golden.py --check
```
Expected: 全绿、ruff 零、golden 无差异。

- [ ] **Step 2: 更新 STATUS.md**

- 「终审 Deferred Minors（24 项，未修）」节标题改为「（24 项，**已于 2026-10-13 清偿**，逐项映射见 `docs/superpowers/plans/2026-10-13-medintel-deferred-minors.md`）」，正文保留原文供追溯。
- 验证基线行的 `194 passed` 更正为实测数字。
- 「当前状态」加一行：24 项 deferred 清偿完成（21 任务、行为修复 14 项、覆盖钉 6 项、文档 2 组）。

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/STATUS.md
git commit -m "docs: close out 24 deferred minors with plan mapping"
```

- [ ] **Step 4: 推送（需用户确认）**

经代理推送（本机直连 github 443 不通）：

```bash
git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 push origin master
```

推送前征得用户确认。
