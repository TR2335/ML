# MedIntel 真实规模优化（P-Scale）实施计划（实施后补录）

> **状态：已实施完毕**（T1–T8，2026-10-12，逐任务 TDD 独立提交）。
> 本文件为实施后补录版：记录任务分解、执行期裁定、Review Focus 测试锚点与提交映射。
> 设计原文（论证过程）：`docs/superpowers/specs/2026-10-12-medintel-real-scale-optimization-design.md`（方案 A 并行增量）。
> 执行方式：用户裁定跳过事前计划评审直接实现（「实现吧」）；红-绿-提交循环 + 全新上下文终审收尾。

**Goal:** 面向 2–3 万出院/入院规模验证方法学与工程链路——G1 规模队列生成、G2 患者级防泄漏切分、G3 自动模型报告卡、G4 PSI+KS 双指标漂移、G5 端到端规模演练；演示路径与 156 基线测试零回归。

**Architecture:** 新增路径与既有脚本共存（`app/cohort.py` 纯标准库生成器 + 3 个新脚本），复用既有指标函数（`binary_metrics`/`evaluate_rows`/`bootstrap_metric` 等）防口径漂移；`app/` serving 纯净约束不变。

**Tech Stack:** Python 3.11+、纯标准库 serving、scikit-learn 仅训练侧、pytest（新增 `slow` marker）、ruff。

## Global Constraints（继承 spec）

- `app/` 零 sklearn/numpy/scipy/lightgbm/optuna（`app/cohort.py` 仅标准库）；scripts/ 允许 numpy。
- scale ∈ [1000, 50000] 默认 25000；seed 默认 20261012；同 (scale, seed) 字节级确定。
- landmark 网格 16 格/遭遇（6h 步进）；`encounter_id` 保持 `enc-<序号>` 格式。
- 门禁阈值口径不变（ECE<0.05、亚组 ≤10% 等）；KS 看效应量不看 p 值（默认阈值 0.1）。
- bootstrap：>50000 行 → 500 次 × ≤20000 行子采样，报告注明 `replicate_subsample`。
- 缺分组键响亮报错，不静默降级行级；报告卡指标只许复用既有函数。
- Windows：`perf_counter` 计时，psutil 可选；ruff line-length 100。

## Review Focus（实现前识别的五类失败 → 测试锚点）

1. **同患者多次住院跨折泄漏**（G2 核心）→ `test_patient_split_never_splits_a_patient`、`test_patient_split_keeps_simultaneous_patients_on_one_side`、`test_patient_split_missing_group_keys_raises`、`test_patient_split_falls_back_to_encounter_id`。
2. **binary_metrics AUPRC O(n²) 在规模行数挂死**（spec 未预见；bootstrap/评估/门禁全依赖）→ `test_binary_metrics_handles_100k_rows`（旧实现 180s 超时即红）+ `test_binary_metrics_auprc_matches_quadratic_reference`（整数前缀和位级等价）。
3. **规模 CSV 21 维映射与 serving `_features_before` 契约分叉**（训练/在线静默不一致）→ `test_load_dataset_csv_matches_features_before`（含缺失行逐位对照 NaN）。
4. **默认 patient 模式响亮报错波及既有无分组键输入**（promote/evaluate 回归）→ 既有 holdout fixture 补 `encounter_id`（test_manage_model 3 处 + test_drift 2 处 + test_evaluation 2 处）+ 全量套件回归。
5. **合成队列比率漂出门禁 sanity 窗口**（演练假失败/假通过）→ `test_cohort_rates_within_spec_windows`（deterioration 按遭遇级事件率口径，见 R-RS-5）+ 演练 `checks.metrics_sanity`。

## 任务分解与提交映射

| 任务 | 内容 | 提交 | 新增测试 |
| --- | --- | --- | --- |
| T1 | `app/cohort.py` + `scripts/generate_scale_data.py`（G1） | `8657956` | test_scale_cohort 6 |
| T2 | `patient_rolling_time_split` + validate/evaluate/promote `--split`（G2） | `068da34` | test_validation 8 + manage 2 + evaluation 2 |
| T3 | `bootstrap_budget`/`max_rows` + AUPRC O(n log n)（G5 性能前置） | `5cdd7b3` | test_validation 2 + evaluation 2 + manage 断言扩展 |
| T4 | `train_model.py --input` 规模 CSV 训练（G5 数据路径） | `7425a01` | test_training 3 |
| T5 | `scripts/report_card.py` 10 章节（G3）+ `decision_curve_table` | `4341123` | test_report_card 5 |
| T6 | `compute_ks` + `drift_reference.sample`（G4） | `0d86592` | test_drift 6 |
| T7 | `scripts/scale_rehearsal.py`（G5）+ monitor 派生特征即时编码 | `aadf07a` | test_scale_rehearsal 1（slow）+ test_drift 1 |
| T8 | 真跑 `--scale 25000` + README/演化文档回填 + 终审 | 本文件所属提交 | — |

基线演进：156 → 194 passed（+38）+ 1 skipped + 2 deselected；全程 ruff 全零。

## 执行期裁定（Plan Rulings）

1. **R-RS-1 患者分配公式**：唯一患者 = `round(scale × 0.82)`，多次患者各恰好 2 次（22% 多次住院与 25000/20500 自洽；spec §2 三个数字的唯一整数解）。
2. **R-RS-2 网格 16 格 = 6..96h**：生命体征 17 条（0..96h，与演示同构），landmark 取 6..96h 保证首格窗口有历史（spec "0..96h → 16 个"的算术歧义落法）。
3. **R-RS-3 dataset 列序**：demo_dataset 契约列序 + `patient_id` 插第 2 列 + `sex/facility_id/department` 尾列（患者级切分与逐院区门禁需要元数据列，"同列序"字面冲突的解法）。
4. **R-RS-4 readmission 快照不注入缺失**：LACE 基线要求字段完整；MAR 缺失仅注入 deterioration 连续体征。
5. **R-RS-5 deterioration sanity 按遭遇级事件率**：阳性 landmark 行率 ≈1%（事件×前 24h 窗格），spec "阳性率 ±3pp"按行级解释必然失败；readmission 行级=患者级口径不变。
6. **R-RS-6 事件时刻与 LOS 解耦**：合成局限（事件可在出院后），报告卡注明"合成数据"；真实数据由 P9 契约约束。
7. **R-RS-7 既有测试 CSV 补 encounter_id**：patient 默认模式的响亮校验需要（7 处 fixture），真实 holdout 本应含分组键。
8. **R-RS-8 `resolve_group_column` 住 `evaluate_model.py`**：validate→evaluate 已有顺向 import，避免循环依赖。
9. **R-RS-9 门禁患者级口径落点**：`run_promotion_gate(split_mode=)` 校验 holdout 分组键 + eval.json 记 `split_mode/group_column`（留出集不切分，口径=分组键完整性 + 记录）。
10. **R-RS-10 `bootstrap_budget` 共享 helper**：validate 与 promote 同一规模规则（>50000 → 500×≤20000）。
11. **R-RS-11 AUPRC O(n log n)**：整数前缀和与旧实现位级一致（对照测试钉住）；100k 行可完成。
12. **R-RS-12 报告卡 model 列 = OOF 复用**：`model_with_news2` 显式 skip 注明（避免二次全量引擎打分；增量列由 evaluate labeled_csv 产出）。
13. **R-RS-13 参考样本上限 2000/特征**：spec 5000 的工件体积修正（~500KB/工件，KS n=2000 足够）。
14. **R-RS-14 KS 效应量固定阈值 0.1**：不引入样本量自适应临界值（spec 裁定）；n≈300 的小批量监控下接近 99% 临界，留意误报（见 Deferred）。
15. **R-RS-15 monitor 派生特征即时编码**：漂移输入为出院快照源字段 CSV 时按 `encode_readmission_payload` 派生（serving R2 同语义；否则 `diagnosis_code_count` 全 missing 误报——演练中实际咬到）。
16. **R-RHR-1 演练补门禁步骤**：spec 通过标准 2 要求门禁但步骤表漏列；作为 gate 组补入（预算 180s，合计 1200→1380s）。
17. **R-RHR-2 漂移自检输入 = holdout 留出段**：训练参考 vs 尾 20% 时间段（同分布不误报即验证通过）。
18. **R-RS-16 `slow` marker 默认执行**：spec "CI 可跑"→ addopts 不排除；1500 冒烟 ~3.5 分钟计入套件。

## 本轮新增 Deferred（未修，留痕）

- 报告卡 `model_with_news2` 列 skip（R-RS-12）：需要三列增益时走 `evaluate_model --input-format labeled_csv`。
- `report` 时间预算组 @25000 实际超出 spec 60s（两目标报告卡含 OOF 打分 + bootstrap；budget_exceeded 留痕不阻断）。
- KS 固定效应量阈值在 n≈300 输入下接近统计临界（R-RS-14）：小批量场景建议调大 `--ks-threshold` 或加大批量。
- 报告卡 `_decile_calibration` 是展示表（非门禁指标）；十分位口径与 `_expected_calibration_error` 固定宽分箱不同源。
- 演练 promote 的 C 选择 CV 见过 holdout 标签（沿用 P6 R4 既有行为，isotonic/fit 仍只用前 80%）。
- 演练失败路径可读到上次成功的 `eval.json`/`cohort_stats.json`（陈旧产物）；总 `passed` 由 `steps_ok` 兜底仍为 False，仅报告字段失真（终审 M2，修复成本=run 起始清理或 mtime 校验）。
- `_drift_reference` 的 lo/hi/proportions 仍按全量矩阵计算（含尾 20% 段）；`sample` 抽样 seed 已接 `--seed`（终审 M1 的 seed 半边已修，train-段口径待裁定）。

## 终审与修复轮（2026-10-12）

- 终审（全新上下文 opus reviewer，diff `96b4140..HEAD`）：**0 Critical / 2 Important / 12 Minor**；全量测试与 ruff 复现通过。
- **Important 已修（唯一修复轮）**：① 报告卡 LACE 基线归一化 `/19` + 阈值 `4/19`（`BASELINE_THRESHOLDS`；修复前 25k 真实产物 ece=5.32 定义失真）+ readmission 报告卡回归测试；② `split_strategy` 披露 fit 窗（前 60% 时间行）与逐折 `test_rows_in_fit_window/Beyond`（25k 实测 50.4% OOF 行落在 fit 段的乐观偏差留痕；门禁/ sanity 用尾 20% holdout 不受影响）。
- **Minor 已修 9 项**：drift sample 接 `--seed`、scale_sanity 患者数核对、run_step 超时硬中断、leakage violations 机读类型稳定、resolve 文案分支、ks_reason 区分空样本、encounter 分支 SystemExit 包装（validate+report_card）、split_strategy 行级时间交错说明、bootstrap_ci 记 seed。
- **Minor 入 deferred 2 项**：陈旧产物（M2）、drift reference 全量口径（M1 后半）。
- 修复轮后基线：**196 passed + 1 skipped + 2 deselected**；ruff 全零。

## 验证与收尾

- 全量 `pytest` 194 passed + 1 skipped + 2 deselected（含 1500 冒烟）；`ruff check` + `format --check` 全零。
- 真跑 `--scale 25000` 演练：见 STATUS.md 记录的关键数字（本地产物 `data/rehearsal_report.json`，`data/scale/` 不入库）。
- 终审：全新上下文 code review 全量 diff → 唯一修复轮。
