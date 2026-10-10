# MedIntel 数据接入设计文档（自助版）

> 配套主计划 `docs/superpowers/plans/2026-09-27-medintel-task-breakdown.md` 的 **P9 阶段**。
> 主计划 P0–P8 的实现**不依赖本文件**；本文件供你在拿到数据后**自行完成接入、再训练与切换**。
>
> 能力状态标注：✅ = 现有代码已支持；🆕 = 主计划 P2–P6 任务落地后可用（按主计划实现的顺序自动具备）。

---

## 0. 总览：你要走的 9 步

```text
你的数据源（院内导出，已去标识）
   │
   ① 整理成 患者文件 + 事件文件（JSON/NDJSON，契约见 §2/§3）
   ② scripts/import_clinical_data.py  dry-run → 复核 → --commit    导入 ✅
   ③ 整理 结局文件（契约见 §4）
   ④ scripts/derive_labels.py   → labeled.csv                      打标签 🆕
   ⑤ scripts/build_features.py  → features.csv                     特征   🆕
   ⑥ scripts/train_model.py --input features.csv → 工件 JSON        训练   🆕
   ⑦ scripts/evaluate_model.py  → model vs news2 vs model_with_news2 评估 🆕
   ⑧ scripts/thresholds.py 反推阈值 → scripts/manage_model.py promote 门禁 🆕
   ⑨ make_golden.py 再生 golden → 复验离在线一致 → 服务切换新工件   🆕
```

每步的命令示例在对应章节；全部跑通即完成主计划的 **M-G 里程碑**。

---

## 1. 前提与责任边界

- **去标识化与治理审批在你/机构侧完成**。系统只接受去标识数据：`display_name` 置空，患者标识用假名或哈希。
- **时间语义**：所有时间用 ISO 8601（例 `2026-03-01T08:30:00Z` 或 `2026-03-01T08:30:00+08:00`）。**不带时区的值按 UTC 解释**——院内本地时间必须先转 UTC 或带偏移，否则全部错 8 小时。
- **幂等**：同一 `tenant_id` 下重复 `event_id` 会被识别为重复并去重，重复投递安全。
- **未来时间戳会被拒绝**（导入与 API 同一校验语义，T2.4 落地后强制）。
- 主键一致性：事件的 `(tenant_id, facility_id, encounter_id, patient_id)` 必须与患者文件中的记录完全一致，否则**整批拒绝**。

---

## 2. 契约一 · 患者文件（`--patients`）

格式：JSON 数组，或 NDJSON（每行一个对象）。编码 UTF-8（允许 BOM）。

| 字段 | 类型 | 必填 | 约束 | 示例 |
|---|---|---|---|---|
| `patient_id` | str | ✅ | 1–128 字符 | `"p-0001"` |
| `encounter_id` | str | ✅ | 1–128 字符 | `"enc-0001"` |
| `display_name` | str | ❌ | ≤256，默认 `""`（必须去标识） | `""` |
| `age` | int | ✅ | 0–120 | `67` |
| `sex` | str | ❌ | `M` / `F` / `O` / `U`，默认 `U` | `"F"` |
| `department` | str | ❌ | 默认 `unknown` | `"cardiology"` |
| `admitted_at` | datetime | ❌ | ISO 8601，默认当前时间 | `"2026-03-01T08:30:00Z"` |
| `status` | str | ❌ | `inpatient` / `discharged` / `cancelled`，默认 `inpatient` | `"inpatient"` |
| `tenant_id` | str | ❌ | 默认 `group-demo` | `"group-demo"` |
| `facility_id` | str | ❌ | 默认 `hospital-01` | `"hospital-01"` |

---

## 3. 契约二 · 事件文件（`--events`）

格式同上（JSON 数组或 NDJSON）。

| 字段 | 类型 | 必填 | 约束 | 示例 |
|---|---|---|---|---|
| `event_id` | str | ❌ | 1–128；**建议用源系统主键**（保证重导幂等），缺省随机 `evt-<10位hex>` | `"lis-88213"` |
| `event_type` | str | ✅ | `vital_signs` / `lab_result` / `admission` / `discharge` / `medication` / `note` | `"vital_signs"` |
| `occurred_at` | datetime | ✅ | ISO 8601，naive 按 UTC | `"2026-03-01T09:00:00Z"` |
| `tenant_id` | str | ❌ | 默认 `group-demo` | `"group-demo"` |
| `facility_id` | str | ❌ | 默认 `hospital-01` | `"hospital-01"` |
| `patient_id` | str | ✅ | 须与患者文件一致 | `"p-0001"` |
| `encounter_id` | str | ✅ | 须与患者文件一致 | `"enc-0001"` |
| `actor` | str | ❌ | 默认 `system` | `"system"` |
| `source_system` | str | ❌ | 血缘可选 | `"LIS"` |
| `source_record_id` | str | ❌ | 血缘可选，≤256 | `"88213"` |
| `unit` | dict[str,str] | ❌ | 计量单位 | `{"heart_rate": "bpm"}` |
| `quality_status` | str | ❌ | `unknown` / `valid` / `suspect` / `invalid` | `"valid"` |
| `payload` | dict | ❌ | 数值载荷，见下表 | `{"heart_rate": 92, "spo2": 96}` |

### payload 数值字段合法范围

超出范围或传布尔值（`true`/`false`）的事件会被拒绝（布尔问题在 T3.3 后返回 422 而非 500）：

| 字段 | 范围 | 字段 | 范围 |
|---|---|---|---|
| `age` | 0–120 | `spo2` | 0–100 |
| `heart_rate` | 20–300 | `wbc` | 0–200 |
| `respiratory_rate` | 4–100 | `creatinine` | 0–50 |
| `systolic_bp` | 40–300 | `lactate` | 0–50 |
| `temperature` | 25–45 | `prior_admissions_30d` | 0–100 |

限制（T4.5-7 落地后强制）：单值 ≤4KB，单事件 payload 总计 ≤64KB。

**示例事件**（合成值）：

```json
{
  "event_id": "lis-88213",
  "event_type": "lab_result",
  "occurred_at": "2026-03-01T09:00:00Z",
  "patient_id": "p-0001",
  "encounter_id": "enc-0001",
  "source_system": "LIS",
  "payload": {"creatinine": 2.1, "lactate": 3.4}
}
```

---

## 4. 契约三 · 结局文件（标签输入，`derive_labels.py`）

格式：CSV，表头固定三列：

```csv
encounter_id,outcome_type,occurred_at
enc-0001,rrt,2026-03-02T03:10:00Z
enc-0007,unplanned_readmission,2026-04-10T11:00:00Z
```

- `outcome_type` ∈ `rrt` | `icu_transfer` | `cardiac_arrest` | `death` | `unplanned_readmission`
- 时间语义与事件一致（naive 按 UTC）。

### 标签定义（写死的口径，与主计划 T5.1 一致）

| 目标 | 定义 | 窗口边界 |
|---|---|---|
| `deterioration_24h` | `index_time` 后 24 小时内发生 `rrt` / `icu_transfer` / `cardiac_arrest` / `death` 任一 | **严格 `< 24h` 算阳性；恰好 24h 整点算阴性** |
| `readmission_30d` | 出院（`discharge` 事件）后 30 天内发生 `unplanned_readmission` | 同理严格 `< 30d` |

- **index_time（landmark）生成**：从 `admitted_at` 起每 24 小时一个 index（`--index-interval-hours 24`，可调），出院后不再生成；用于恶化目标。再入院目标的 index = `discharge` 事件时刻。
- **防泄漏**：特征窗口必须**严格早于** `index_time`（`build_features.py` 强制）。

---

## 5. 契约四 · 出院快照（P8/T8.1，再入院目标特征）

再入院模型的特征来自出院时快照，字段（T8.1 落地后导入校验强制，缺字段的出院记录会被拒并给出字段级错误）：

| 字段 | 类型 | 含义 | 示例 |
|---|---|---|---|
| `discharge_destination` | str | 出院去向 | `"home"` |
| `los_days` | float | 住院天数 | `5.2` |
| `diagnosis_codes` | list[str] | 出院诊断编码 | `["I50.9", "N17.9"]` |
| `comorbidity_index` | float | 合并症指数 | `2.0` |
| `prior_utilization_12m` | int | 近 12 个月利用次数 | `1` |
| `ed_visits_6m` | int | 近 6 个月急诊次数 | `2` |

---

## 6. 导入命令（✅ 现在就能用）

```bash
# 1) dry-run：只校验、打印 summary，不写库
python scripts/import_clinical_data.py --patients patients.json --events events.json

# 2) 复核 summary 后正式导入
python scripts/import_clinical_data.py --patients patients.json --events events.json --commit
```

summary 输出字段：`patients`、`events`、`commit`、`event_types`；`--commit` 后追加 `accepted_events`、`duplicate_events`（T2.4 落地后中途失败会附 partial summary：accepted/failed 计数 + 最后成功键）。

先用演示数据熟悉流程：`python scripts/generate_demo_data.py --reset`。

---

## 7. 标签 → 特征 → 训练 → 评估 → 门禁（🆕 按主计划 P5/P6 实现后可用）

```bash
# ④ 打标签（阳性率异常时先查时间时区与结局覆盖）
python scripts/derive_labels.py \
  --events events.json --outcomes outcomes.csv \
  --target deterioration_24h --index-interval-hours 24 \
  --output labeled.csv

# ⑤ 特征（七元组 ~70 维 + news2_score，顺序由 app/features.py 的 FEATURE_ORDER 固定）
python scripts/build_features.py --input labeled.csv --output features.csv

# ⑥ 训练（--input 为本设计约定的接口，T6.1 实现时以此为准；--rows 为合成演示模式）
python scripts/train_model.py \
  --input features.csv --target-id deterioration_24h \
  --output data/model.json

# ⑦ 评估：model / news2 / model_with_news2 三列指标
python scripts/evaluate_model.py --input-format labeled_csv \
  --input features.csv --artifact data/model.json

# ⑧ 反推阈值 + promote 门禁
python scripts/thresholds.py --target-sensitivity 0.90 --artifact data/model.json
python scripts/manage_model.py promote --target-id deterioration_24h --version <version>
#   门禁：AUPRC 95% CI 下界 ≥ NEWS2，且 ECE < 0.05；不达标拒绝并打印差距

# ⑨ 再生 golden dataset 并复验离在线一致
python scripts/make_golden.py --check        # 提示需再生时去掉 --check 重新生成
pytest tests/test_serving_consistency.py -q
```

再入院目标把 `--target` 换成 `readmission_30d`，特征集用 §5 的出院快照字段，门禁换成"打赢 LACE"（T8.3）。

---

## 8. 工件接入服务

- `manage_model.py promote` 按 target/version 保存工件并生成 `manifest.json`（含 SHA-256）。
- 工件 JSON 向后兼容：`weights/bias/feature_names/thresholds/model_version` 永不删除，新增 `preprocess`、`calibration`、`feature_order`、`trees` 块。

| 环境变量 | 作用 | 建议值（真实数据） |
|---|---|---|
| `APP_ENV` | 运行环境 | `prod`（dev 才允许演示令牌） |
| `STORAGE_BACKEND` | `sqlite` / `postgres` | 生产用 `postgres` |
| `DATABASE_URL` | 数据库连接（非 dev 不允许默认值，T3.5） | 你的实例 |
| `EVENT_BACKEND` | `sync` / `redis` | 高并发用 `redis` |
| `REDIS_URL` / `REDIS_STREAM` | Redis 连接与流名 | 你的实例 |
| `MODEL_SIGNATURE_REQUIRED` | 非 dev 默认开（T3.1），校验 SHA-256 | `true` |
| `MODEL_SIGNATURE_PATH` | manifest 路径 | promote 产物 |
| `MODEL_FALLBACK_ENABLED` | 加载失败是否降级 | 生产保持 `false`（fail-closed） |
| `DEMO_AUTH_ENABLED` | 演示令牌（仅 `APP_ENV=dev` 有效） | `false` |

---

## 9. 数据量分层与算法选型

| 阳性样本量 | 版本 | 你要做的 |
|---|---|---|
| < 500 | 标注不足 | `scripts/active_label.py` 出 top-N 待标注清单（按 `\|p-0.5\|` 升序），**人工标注**后回灌重训 |
| 500 – 5,000 | P6 逻辑回归（默认） | 走 §7 全流程 |
| > 5,000 | P7 LightGBM | `train_model.py --algorithm lightgbm`（单调约束 + Optuna），门禁全量通过后 promote |

全部跑通后，按你的真实阳性率复核 5,000 分界与 `--target-sensitivity 0.90`（主计划 T9.3），必要时修订主计划。

---

## 10. 验收清单（全部勾掉 = 完成 M-G）

- [ ] dry-run 零失败，summary 计数与源数据核对一致
- [ ] 所有时间戳带时区或已转 UTC；无未来时间戳
- [ ] `derive_labels.py` 阳性率落在合理区间并**记录下来**（demo 口径 5–15%）
- [ ] 特征窗口确认严格早于 index_time（无泄漏）
- [ ] 评估报告含 `model` / `news2` / `model_with_news2` 三列，附增益数字
- [ ] promote 门禁通过，`model-registry/<version>/eval.json` 已生成
- [ ] `make_golden.py --check` 通过；`pytest tests/test_serving_consistency.py` 全绿
- [ ] 服务加载新工件，抽查若干病例：服务分数 == 离线分数
- [ ] `docs/model-card.md` 同步新版本数据口径与限制

---

## 11. 常见坑

1. **时区**：最常见的全错来源。导入前检查无裸 `2026-03-01 08:30`（无时区）格式。
2. **主键对不上**：事件与患者文件的四元组不一致 → 整批拒绝，先对齐再导。
3. **重复导入**：靠 `event_id` 幂等——换了随机 ID 会产生重复事件，**务必用源系统主键**。
4. **阳性率离谱**：先查结局文件时区、再查窗口边界（严格 `<24h`）、最后查 index 生成间隔。
5. **离在线分数分叉**：改了工件 `preprocess`/`calibration` 却没再生 golden → `make_golden.py --check` 必进流程。
6. **门禁差一点**：先看 NEWS2 对比列——若 model 无增益，回查特征质量与标签口径，而不是直接降门禁。
