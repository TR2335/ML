# MedIntel 终版状态（2026-10-13 更新，deferred 清偿轮）

> 新终端会话从这里恢复上下文。计划原文：`docs/superpowers/plans/2026-10-12-medintel-real-scale-optimization.md`（P-Scale 补录版）、`2026-10-10-medintel-p7.md`、`2026-10-10-medintel-p8.md`、`2026-09-28-medintel-p6.md`（及 P0–P5 同目录）；设计 spec：`docs/superpowers/specs/2026-10-12-medintel-real-scale-optimization-design.md`。

## 当前状态

- **P-Scale（真实规模优化 G1–G5）已于 2026-10-12 实施完毕**：T1–T8 逐任务 TDD 独立提交（`8657956..HEAD`），18 项执行期裁定（R-RS-1..16、R-RHR-1..2）与任务映射见计划补录文档。
- **25000 规模端到端演练真跑 passed=True**（936.6s / 预算 1380s 内）：泄漏 0 违规；双门禁全过（ECE 0.0007 / 0.0082）；事件率 4.1%、再入院 12.24%；40 万行 deterioration + 3 院区 45/35/20；holdout 80000/5000；超预算留痕 `evaluate`（132s>120s）与 `report`（263s>60s，见 deferred）。
- 验证基线：`pytest` → **223 passed + 1 skipped + 2 deselected**（含 1500 规模冒烟 `@slow`，默认执行）；`ruff` 全零；`make_golden --check` 一致；`app/` serving 纯净（新增 `app/cohort.py` 仅标准库）。
- **24 项终审 deferred minors 已于 2026-10-13 全部清偿**（21 任务，逐项映射见 `docs/superpowers/plans/2026-10-13-medintel-deferred-minors.md`）。行为修复要点：`binary_metrics` AUPRC 改平局分组口径（恒分→prevalence，sklearn 语义）；**facility 门禁规则改同院区基线比较且阳性下限 5→10**（AUPRC 依赖患病率，跨总体比较统计无效——平局修复暴露的潜伏缺陷）；阈值撞档改拒绝；CV 零阳性折跳过；live readmission 钉首个出院事件（worker+api）；recalibrate 严格 0/1 + gate-input 预检；门禁输入错误统一拒绝语；`top_factors` 的 `__missing` 展示为「基特征缺失」；`training_rows` 只记训练段、校准块增 `n_samples/n_positives`（golden 已重冻结）。
- **P6 / P7 / P8 状态不变**：56 提交已推送（2026-10-10）；三轮终审结论与 24 项 deferred minors 见下文（仍然有效）。
- P-Scale 相对 P8 的接口变化：`validate`/`evaluate`/`promote` 新增 `--split`（默认 **patient**，缺分组键响亮报错——既有 holdout CSV 需含 `patient_id` 或 `encounter_id`）；`train_model --input` 吃规模 CSV；`monitor_drift` 输出 `ks/ks_reason` 且对出院源字段 CSV 即时派生；工件 `drift_reference` 新增 `sample`（≤2000/特征）。

## 待办

1. ~~推送（56 提交）~~ 已完成（2026-10-10）。
2. ~~P-Scale 提交推送~~ 已完成（2026-10-13，12 提交推送至 origin/master，`1804254..e6d8252`；本机需经 127.0.0.1:7897 代理）。
3. **P9 真实数据自助接入**：用户取数后按 `docs/data-integration-design.md` 执行（出院快照 6 字段契约已定：`discharge_destination/los_days/diagnosis_codes/comorbidity_index/prior_utilization_12m/ed_visits_6m`）；真实数据下先跑 `scale_rehearsal` 同款链路（`generate_scale_data` 换成真实导入），患者级切分与报告卡即刻可用。

## 关键运行时事实

- `readmission_30d` 仅出院事件触发单次评分（`as_of` 钉首个出院事件）；demo 种子不再预置 readmission 分数。
- promote 门禁 = bootstrap CI 下界 ≥ 基线（deterioration→NEWS2、readmission→LACE）+ ECE<0.05 + 亚组灵敏度差≤10%（age 可用时，全体组 max−min）+ 逐院区 AUPRC≥**同院区**基线点估计（≥2 院区、每合格院区 ≥20 行且 ≥10 阳性；全跳过 → unchecked/skipped）+ 决策曲线平均增益≥0；不可用规则 `skipped` 不判失败，报告在 eval.json `rules`。AUPRC 为平局分组口径（恒分输入 = prevalence）。
- 校准：readmission 默认真 isotonic（cal 段出样本 8 档）；**demo deterioration 触发 R-P7-5a 回退 Platt**（in-sample PAV 仅 2 档，`fallback_reason=isotonic_resolution`）——真实数据块数充足时自动走 isotonic。
- LightGBM 为 opt-in 挑战者：`scripts/train_model.py --algorithm lightgbm --trials N`；树工件校验严格（缺 trees/未知 algorithm/子索引越界/内部负索引/环 → 载入期响亮拒绝）。
- 漂移：训练写 `drift_reference`；`scripts/monitor_drift.py`（PSI>0.2 → JSON alert + exit 1，输入 CSV 须含完整 `feature_order` 架构含派生列）；`scripts/recalibrate.py` 只动 `calibration` 块、临时工件过门禁才回写（真实数据务必 `--gate-input` 独立留出集）。
- 阈值阶梯手算基准 {0.10, 0.20, 0.30}（计划原稿0.25/0.35 是算术错误，已修正）。

## 终审 Deferred Minors（24 项，已于 2026-10-13 清偿；逐项映射见 `docs/superpowers/plans/2026-10-13-medintel-deferred-minors.md`，以下原文保留供追溯）

**P6（10）**：`training_rows` 记全量非60%；阈值撞档步进后不复检 TPR；`TARGET_THRESHOLDS`/`DEFAULT_THRESHOLDS` 双份硬编码；CV 无阳性测试折 auprc 计0而非跳过；`top_factors` 可能露 `__missing` 裸工程名；active_label 输入含 score/uncertainty 列时重复输出；门禁输入类错误（缺 label 列/非数字/文件不存在）走 traceback 非统一拒绝语；`make_golden.DEFAULT_ARTIFACTS` 与 RiskEngine 默认路径隐式耦合；**`binary_metrics` 恒分 AUPRC 经 (score,label) 元组平局得 1.0（P5 遗留怪癖，修复需同步门禁 fixture）**；校准审计链仅 `fitted_on`。

**P8（6）**：`READMISSION_POSITIVE_RATE` 成死常量；`data-integration-design` summary 字段清单漏 `discharge_events`；旧 vitals 形态 readmission 工件 promote 白名单半边（`tuple(FEATURES)`）无回归测试；多出院事件 live（触发时刻）vs 批路径（首出院）语义分叉与契约措辞未对齐；`DISCHARGE_SNAPSHOT_FIELDS` 与 `READMISSION_PAYLOAD_KEYS` 双常量无交叉相等测试；门禁 LACE 字段缺失 SystemExit 无集成测试。

**P7（8）**：门禁 `float(age)` 对非数值抛裸 ValueError、部分空 age 时 "age column absent" 措辞误导；Optuna `subsample` 惰性参数（缺 `subsample_freq>0`）；`per_facility_auprc` 全部院区被跳过仍 `checked=True`、`subgroup_sensitivity_diff` ≥3 组只比较前两组；monitor_drift 输入 schema 要求未写入 architecture.md；recalibrate 拒绝半场用全零标签在输入校验处即拒（度量型拒绝偏弱）；**isotonic 的 Brier 不劣化无断言**（建议补 readmission 变体）；R-P7-5a 回退 docs 未说明（commit 标题对默认目标言过其实）；`recalibrate` 的 `int(label)` 接受 "2" 等非 0/1 值。

## 执行期裁定要览（完整计划级裁定见各计划的 Plan Rulings 节）

- master 直做 = 用户本人裁定；OUTCOME_TYPES 加 `unplanned_readmission`；CV 单类折跳过；readmission 出院时间 %7 天 LOS 抖动（7 时间戳）；门禁 fixture 阳性率 25%；CI 训练步前移 pytest 之前；golden 篡改目标 median→mean（fixture 零缺失行）；P8 移除 lifespan readmission 种子（种子 as_of 会永久挡触发写入）；LACE L 分箱 (0,1]/(1,4)/[4,7)/≥7（已获 Austin 2010 对照确认）；`_cell` JSON 序列化 list；labeled_csv+readmission 显式拒绝；P7 走树打分并入 Task1；容差 5.1e-5（4 位舍入契约，未舍入1e-6由 `tree_raw_margin` 断言承担）；`tree_debug` 单钩子；**R-P7-5a isotonic 分辨率回退 Platt**；imbalance fixture bias −4.128、facility 反例基线 0.3；recalibrate 测试 readmission 化+去 age 列；PSI 0.005 平滑 + 同源冒烟。
