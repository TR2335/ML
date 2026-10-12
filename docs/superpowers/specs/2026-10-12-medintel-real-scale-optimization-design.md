# MedIntel 真实规模优化设计（2–3 万出院/入院规模）

> 状态：**待评审**（spec，尚未实施）
> 日期：2026-10-12
> 对应路线图：`docs/project-evolution.md` §5.1「本轮优化（进行中）」
> 前置/后继：本设计是 P9（真实数据自助接入）的前置必修；实施前需按 `superpowers:writing-plans` 生成实施计划

---

## 1. 背景与目标

### 1.1 背景

项目当前是"研究与生产前验证骨架"，所有离线 ML 链路都跑在 **18 个演示患者 / 2500 行训练数据** 的玩具规模上（`app/synthetic.py:20`、`scripts/train_model.py:410`）。在真实项目里，2–3 万条出院/入院量级是单中心一年半到两年的常规体量，这个规模才会暴露三类问题：

1. **方法学问题**：同患者多条 landmark 行跨折泄漏，在小样本里看不出来，在真实规模下会让指标虚高；
2. **工程问题**：逐小时 landmark 网格 × 2.5 万遭遇 = 240 万训练行，CSV、bootstrap、评估链路全部需要重新核算复杂度；
3. **交付问题**：门禁指标散落在多次命令行输出里，没有一份随模型走的、给人看的自动报告卡。

### 1.2 目标（本设计完成后的验收对象）

| # | 目标 | 验收方式 |
| --- | --- | --- |
| G1 | 可一键生成 2–3 万规模的合成出院/入院队列（含结局、landmark 数据集） | `scripts/generate_scale_data.py --scale 25000` 产出确定性 CSV |
| G2 | 评估与 promote 门禁切换到**患者级时间切分**，同患者行不跨折 | 每折 train/test 患者集合不相交的硬断言 + 测试 |
| G3 | 自动产出**模型报告卡**（md + json），覆盖数据画像、指标 CI、基线对比、亚组/院区、决策曲线 | `scripts/report_card.py` 单命令生成 |
| G4 | 漂移监控在 PSI 之外增加 **KS 检验**，双指标告警 | `scripts/monitor_drift.py` 输出 `psi` + `ks` |
| G5 | 端到端**规模演练**一次跑通并在时间预算内完成，产出机读演练报告 | `scripts/scale_rehearsal.py` → `data/rehearsal_report.json` |

### 1.3 非目标（明确不做）

- 不接入真实院内数据（P9 范畴，按 `docs/data-integration-design.md` 另行设计）
- 不改门禁规则的阈值口径（ECE<0.05、亚组灵敏度差 ≤10% 等维持 P6–P7 裁定）
- 不做分布式/多机训练；不引入新存储后端；不做 LightGBM 超参在大规模下的系统调优（沿用既有 optuna 预算）
- 不动 serving：`app/` 内 sklearn/numpy/scipy 零进入的纯净约束不变
- FHIR/OIDC/Prometheus/前瞻性评估等生产化项均不在本设计内

---

## 2. 规模假设与数据画像

`--scale N`，默认 **25000**（允许 20000–30000）。按下表参数生成，全部由 `--seed`（默认 20261012）确定：

| 维度 | 设定 | 依据 |
| --- | --- | --- |
| 遭遇（出院/入院记录） | 25,000 条 | 用户指定 2–3 万区间取中 |
| 时间跨度 | 540 天（约 18 个月） | 滚动时间切分需要 ≥3 个可分辨时间块 |
| 唯一患者 | 约 20,500 人，其中 ~22% 有 2–4 次住院 | 患者级切分必须"有牙齿"——同患者多遭遇才测得出泄漏 |
| 院区 | 3 个（hospital-01/02/03，占比 45%/35%/20%） | 逐院区 AUPRC 门禁（Ruling R-P7-7）要求 ≥2 个合格院区 |
| 科室 | 6 个（呼吸/心内/消化/神内/普外/综合） | 亚组灵敏度门禁的分层来源 |
| 30 天非计划再入院率 | **12%**（患者级） | 综合内科文献常见区间 10–15% |
| 24h 恶化事件率（RRT/非计划 ICU/心跳骤停/死亡复合） | **4%**（遭遇级） | 与现有结局类型集合一致（`derive_labels.OUTCOME_TYPES`） |
| 连续特征缺失率 | 5%–12%，且随风险升高而升高（MAR 倾向） | 真实项目的数据质量画像；现有中位数填补管线可消化 |
| 再入院标签信号 | prior_admissions_30d、年龄、合并症计数为主要贡献因子 | 保证规模演练时 AUPRC 可过 LACE 基线 |
| 恶化标签信号 | 七元组生命体征 + 时序漂移（入院后逐段恶化轨迹） | 保证可过 NEWS2 基线 |

**索引网格（关键工程决策）**：

- `deterioration_24h`：**每 6 小时一个 landmark**（0..96h → 16 个/遭遇 ≈ **40 万行**）。逐小时网格会产出 240 万行，CSV 即将达 GB 级、bootstrap 不可行；6h 网格保留 24h horizon 的捕捉能力（horizon=24h > cadence=6h，任一窗口内事件必被至少一个 landmark 捕获），行数可控。
- `readmission_30d`：出院时刻单行/遭遇 ≈ **2.5 万行**（与现有"仅出院触发单次评分"语义一致）。
- 演示路径（`demo_patients_with_outcomes`、现有 golden 数据集、156 个测试）**完全不动**，规模路径是并行新增。

---

## 3. 方案选型

### 方案 A（推荐）：并行增量——规模路径与既有脚本共存，逐件 TDD

新增 `generate_scale_data.py` 与 `scale_rehearsal.py`，`validate.py`/`monitor_drift.py` 加新函数与新参数，`report_card.py` 独立新增；复用 `evaluate_model.py` 的 `binary_metrics`、`app/features.build_landmarks` 等既有纯函数。

- 优点：每件独立可测、独立提交，符合项目 WBS→计划→TDD→终审方法论；演示路径零回归风险；不碰 serving。
- 缺点：脚本数量继续增多（scripts/ 约 16→18 个），但每个仍是单一职责。

### 方案 B：管线重构——抽 `pipeline/` 包统一编排

把 derive→build→train→validate→evaluate 收编成一个带配置的管线模块，规模与演示共用一条代码路径。

- 优点：长期最干净。
- 缺点：触碰 7 个已过终审的脚本的接口，回归面大；本轮目标是"验证规模下方法学成立"，不是重构；与 YAGNI 冲突。

### 方案 C：最小化——只做规模生成 + 患者级切分

报告卡和 KS 留到下轮。

- 缺点：`project-evolution.md` §5.1 明确把四项列为同一轮；报告卡是规模演练的交付物，KS 是漂移双检的组成部分，拆开会让演练报告不完整。

**裁定：方案 A。** 理由：项目演化史证明"小步、独立、可终审"的增量路径在本仓库跑得通（P5–P7 全部如此），且方案 A 天然把 G1–G5 拆成 5 个可独立验收的任务。

---

## 4. 设计分节

### 4.1 规模化队列生成（G1）

**新增 `scripts/generate_scale_data.py`**，核心生成函数放 `app/synthetic.py` 同层的新模块 **`app/cohort.py`**（纯标准库：`random`/`datetime`/`csv`，保持 app/ 零三方依赖；它不含服务逻辑，与 `app/synthetic.py` 同性质）。

```
python scripts/generate_scale_data.py \
  --scale 25000 --seed 20261012 --horizon-days 540 \
  --out-dir data/scale
```

产出（`data/scale/`）：

| 文件 | 行数（25k 默认） | 内容 |
| --- | --- | --- |
| `encounters.csv` | 25,000 | patient_id、encounter_id、入出院时间、院区、科室、年龄、性别 |
| `labels_deterioration.csv` | ~400,000 | encounter_id、patient_id、index_time、label（`derive_labels` 输出 + patient_id 列） |
| `labels_readmission.csv` | ~25,000 | 出院时刻单行，同上 |
| `dataset_deterioration.csv` | ~400,000 | 七元组 landmark 特征 + label（与 `demo_dataset.csv` 同列序） |
| `dataset_readmission.csv` | ~25,000 | 出院快照特征 + label |
| `cohort_stats.json` | — | 规模、阳性率、缺失率、院区/科室分布（供报告卡与演练门禁引用） |

**设计要点**：

1. **流式按遭遇生成**：逐遭遇构造其串行生命体征与结局，构建完 landmark 即写盘丢弃，内存峰值保持在单遭遇 + 写缓冲级别（≤ 数百 MB），避免 40 万条 pydantic `ClinicalEvent` 全量驻留。
2. **确定性**：同一 `(scale, seed)` 产出字节级一致；`patient_id`/`encounter_id` 保持 `enc-<序号>` 既有格式（`train_model.py:85` 等处按 `-` 切分取序号），时间序由独立 `index_time`/`admitted_at` 列承载。
3. **复用既有纯函数**：landmark 特征构建走 `app/features.build_landmarks`，NEWS2/LACE 基线走 `scripts/baselines.py`，口径与演示链路同源。
4. **标签语义不变**：**按遭遇分批**调用 `scripts/derive_labels.derive_labels()`（importable 函数，单批事件驻留内存，与要点 1 的流式约束一致），horizon 与 `OUTCOME_TYPES` 集合零改动。
5. **越界防护**：`scale` 不在 [1000, 50000] 或索引网格参数非法 → `SystemExit` 响亮报错，不静默钳制。

### 4.2 患者级时间切分防泄漏（G2）

**`scripts/validate.py` 新增 `patient_rolling_time_split(rows, n_splits, group_column="patient_id")`**：

1. 每行归入其 `patient_id` 组（缺 `patient_id` 回退 `encounter_id`；两者皆缺 → **响亮报错**，不静默降级为行级切分）；
2. 以**患者首条 index_time** 为患者时点排序，患者组切成 n_splits+1 个连续块——沿用"时间戳不跨折"裁定：同一患者时点的多个患者不拆块；
3. expanding window：折 i 训练块 0..i-1、测试块 i；
4. **返回前自证**：每折 train/test 患者集合交集必须为空，否则 `AssertionError`（把不变量做成代码而非约定）。

**接入点**：

- `validate.py` CLI 增加 `--split patient|encounter`（默认 `patient`；分组键解析顺序 patient_id → encounter_id → 皆缺报错，见上文第 1 条）；`encounter` 模式保留旧行为供 golden/回归对照；
- `evaluate_model.py` 增加同名参数并透传；
- promote 门禁链路（`manage_model.py promote` 触发的验证）默认走 patient 模式——本设计的门禁数字一律以患者级切分为准。

**注意**：`readmission_30d` 同样受益——同一患者多次出院的行也不得跨折；两目标共用同一函数。

### 4.3 自动模型报告卡（G3）

**新增 `scripts/report_card.py`**：

```
python scripts/report_card.py \
  --input data/scale/dataset_readmission.csv \
  --target-id readmission_30d --model data/readmission_model.json \
  --output data/report_card_readmission   # 产出 .md 与 .json
```

**固定章节（JSON 中以 `sections` 数组记录，缺数据章节标注 skip 原因，不静默省略——与"门禁规则不可用即跳过并注明"同哲学）**：

1. **数据画像**：行数、患者数、遭遇数、院区/科室分布、阳性率、逐特征缺失率（读 `cohort_stats.json` 或直接统计输入 CSV）；
2. **切分策略**：patient/encounter、n_splits、各折 train/test 行数与患者数；
3. **判别性能**：OOF AUPRC/AUROC + bootstrap 95% CI（复用 `validate.bootstrap_metric`），逐折明细；
4. **校准**：ECE、Brier、十分位校准表（复用 `train_model.py` 的分箱逻辑）；
5. **基线对比**：NEWS2（deterioration）/ LACE（readmission）三列对比（复用 `evaluate_model.py`）；
6. **亚组与院区**：亚组灵敏度差、逐院区 AUPRC 表（复用 `validate.subgroup_sensitivity_diff` / `per_facility_auprc`）；
7. **决策曲线**：平均增益 + 逐阈值净获益表（复用 `validate.decision_curve_mean_gain`）；
8. **阈值表**：灵敏度阶梯各档的阈值/灵敏度/特异度/阳性预测值；
9. **漂移参考摘要**：`drift_reference` 各特征的 lo/hi/缺失率 + 参考样本数量；
10. **运行元数据**：seed、git commit、包版本、生成时间、耗时。

**硬约束**：所有指标只允许 import 既有的 `binary_metrics`/`evaluate_rows`/`bootstrap_metric` 等函数——**禁止在报告卡里重算一遍指标**，防止口径漂移。md 面向人（临床/评审读），json 面向机读（后续可挂进 `manage_model.py promote` 的 manifest，本期仅落盘，不改 promote 接口）。

### 4.4 漂移监控 KS 检验（G4）

**`scripts/monitor_drift.py` 新增 `compute_ks(reference_sample, values) -> float | None`**（纯 Python，两路排序求 `max|F₁−F₂|`；规模上参考样本 ≤5000/特征，复杂度可忽略）：

- **工件侧**：`train_model.py` 写 `drift_reference` 时，每特征额外存 `sample`——从训练分布按 `--seed` 固定抽样的 ≤5000 个值（存工件体积增量 ~50KB/目标，可接受）；
- **向后兼容**：旧工件无 `sample` → `ks=None` 并在报告中注明 `"ks": null, "ks_reason": "no reference sample"`（跳过不判失败）；
- **告警逻辑**：`alert = psi > 0.2 OR ks > KS_THRESHOLD`（`KS_THRESHOLD` 默认 **0.1**，CLI 可调）；输出 JSON 中 `features[f] = {psi, ks, alert}`；
- 不引入 scipy：KS 统计量是纯排序比较；p 值不纳入（样本量大时 p 值必然显著，工程上只看效应量——此为本轮裁定，报告卡中注明）。

### 4.5 端到端规模演练（G5）

**新增 `scripts/scale_rehearsal.py`**，按序编排并逐段计时（`time.perf_counter`；内存经 `psutil` 可选记录，未安装则跳过并注明）：

```
python scripts/scale_rehearsal.py --scale 25000 --seed 20261012
# → data/rehearsal_report.json（机读）+ data/rehearsal_report.md（人读摘要）
```

**步骤与时间预算**（开发机参考值，超预算即 `budget_exceeded` 标记，不算失败但必须留痕）：

| # | 步骤 | 预算 |
| --- | --- | --- |
| 1 | 生成队列 + 特征数据集（两目标） | ≤ 180s |
| 2 | 训练 deterioration 模型（logistic + 时间 CV 选 C） | ≤ 300s |
| 3 | 训练 readmission 模型 | ≤ 180s |
| 4 | 患者级滚动切分验证 + bootstrap CI（两目标） | ≤ 300s |
| 5 | 评估 + 基线对比（两目标） | ≤ 120s |
| 6 | 报告卡生成（两目标） | ≤ 60s |
| 7 | 漂移自检（训练参考 vs 留出段，两目标） | ≤ 60s |
| **合计** | | **≤ 20 min** |

**bootstrap 规模化裁定**（当前实现 O(n_bootstrap × n_rows) 纯 Python，40 万行 × 1000 次 = 数亿次操作，不可行）：

- `n_rows > 50,000` 时每次 replicate 对 OOF 行**子采样至 ≤20,000 行**（有放回），`n_bootstrap` 降至 500；
- 方法学不变：仍是 percentile bootstrap 对 OOF 预测重采样；样本量充足时子采样对 CI 宽度影响可忽略——**必须在报告卡 bootstrap 章节注明 `replicate_subsample` 参数**；
- 实现允许在 scripts/ 使用 numpy（`[train]` 依赖已有），`app/` 仍然禁止。

**通过标准（全部写入 `rehearsal_report.json`）**：

1. **泄漏断言**：每折患者交集为空，违规数 = 0；
2. **门禁**：两目标的 promote 门禁规则全过，或按"不可用即跳过并注明"留痕（`rules` 数组完整）；
3. **指标 sanity**：AUPRC ≥ 对应基线（bootstrap 下界口径）、ECE < 0.05、阳性率落在第 2 节设定值 ±3 个百分点；
4. **规模 sanity**：数据行数、患者数、院区分布与设定一致；
5. **时间预算**：超项列入 `budget_exceeded`（告警不阻断）。

---

## 5. 数据流与产物

```mermaid
flowchart LR
  A[generate_scale_data.py<br/>--scale 25000] --> B["data/scale/*.csv<br/>cohort_stats.json"]
  B --> C[train_model.py<br/>两目标 + drift_reference.sample"]
  C --> D[validate.py<br/>--split patient + bootstrap CI]
  D --> E[evaluate_model.py<br/>基线三列对比]
  E --> F[report_card.py<br/>.md + .json]
  C --> G[monitor_drift.py<br/>PSI + KS]
  A & C & D & E & F & G --> H[scale_rehearsal.py<br/>rehearsal_report.json]
```

演示链路（`generate_demo_data.py` → 既有 156 测试 → golden）不经过以上任何节点，互不干扰。

## 6. 门禁与通过标准（本设计自身）

- 五个目标 G1–G5 全部有测试覆盖（见 §7）；
- 既有 `pytest`（156 passed + 1 skipped + 2 deselected）与 `ruff` 全零不受影响；
- 规模演练在默认 `--scale 25000` 下走完全部通过标准；
- `app/` 依赖纯净：新增 `app/cohort.py` 仅标准库。

## 7. 测试策略（TDD，逐任务红绿）

| 模块 | 关键测试 |
| --- | --- |
| `app/cohort.py` | 同 seed 字节级确定性；阳性率/院区/患者多次住院比例落在设定区间；越界参数响亮失败；缺失率区间 |
| `generate_scale_data.py` | 列序与 `demo_dataset.csv` 一致；行数 = 遭遇 × 网格；patient_id 全程存在 |
| `patient_rolling_time_split` | 每折患者交集为空（核心不变量）；患者时点不跨折；时间序 train 早于 test；缺 patient_id 与 encounter_id → 报错；encounter 模式回归不变 |
| `report_card.py` | 10 个章节齐全或注明 skip；指标数值 == 直接调 `binary_metrics` 的值（防口径漂移）；json/md 双产物 |
| `monitor_drift.compute_ks` | 与手算样例一致（两组已知分布）；无 sample → None + reason；psi/ks 混合告警真值表 |
| `scale_rehearsal.py` | `--scale 1500` 冒烟全通过（CI 可跑，标记 slow）；budget 字段与耗时字段齐全 |
| bootstrap 子采样 | 子采样模式下 CI 覆盖率仍合理（对照小样本全量模式的粗检） |

## 8. 风险与错误处理

| 风险 | 处置 |
| --- | --- |
| 40 万行 CSV 磁盘/IO（约 60–80MB） | 流式写；演练前检查 `data/` 剩余空间，不足响亮报错 |
| 演练时长不可控（笔记本差异） | 时间预算超项标记不阻断；CI 冒烟用 `--scale 1500` |
| 合成信号过强导致指标失真乐观 | §2 信号因子 + 噪声与缺失已设上限；报告卡注明"合成数据，指标仅验证管线方法学，不代表真实性能" |
| patient 切分改变指标口径导致门禁历史数字不可比 | 报告卡与 rehearsal 明确标注 split 模式；旧 encounter 数字保留作对照章节 |
| `encounter_id` 时间编码被下游正则依赖（如 `train_model.py:85` 按 `-` 切分取序号） | 新 ID 保持 `enc-<序号>` 格式（§4.1 要点 2），时间序由独立时间列承载 |
| Windows 环境（本仓库在 win32 开发） | 不用 `resource`/`getrusage`；计时用 `perf_counter`，内存 psutil 可选 |

## 9. 实施顺序建议（供 writing-plans 拆解）

1. **T1** `app/cohort.py` + `generate_scale_data.py`（G1，含数据画像测试）
2. **T2** `patient_rolling_time_split` + CLI 接入（G2，泄漏断言测试）
3. **T3** bootstrap 子采样裁定（性能前置，G5 依赖）
4. **T4** `report_card.py`（G3，复用既有指标函数）
5. **T5** `drift_reference.sample` + `compute_ks`（G4）
6. **T6** `scale_rehearsal.py` 编排 + 通过标准（G5）
7. **T7** 真跑 `--scale 25000` 演练、回填 README/演化文档、终审

每任务：写测试 → 实现 → 独立提交 → 下一任务；全程不碰 `app/` serving 行为与演示链路。

## 10. 关键裁定索引（与既有 Ruling 一致）

- 沿用：时间戳不跨折、门禁不可用即跳过并注明、serving 纯净、seed 确定性、golden 钉一致性；
- 本轮新增裁定：**索引网格 6h**（性能 vs 捕捉能力）、**bootstrap 子采样 ≤20k**（CI 方法学不变）、**KS 看效应量不看 p 值**（大样本必然显著）、**报告卡只许复用指标函数**（防口径漂移）、**缺分组键响亮报错而非静默降级**（防假阴性通过）。
