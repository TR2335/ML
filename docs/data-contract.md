# 数据契约

## 临床事件

所有输入统一为事件信封：

```json
{
  "event_id": "evt-1001",
  "event_type": "vital_signs",
  "occurred_at": "2026-09-09T08:30:00Z",
  "tenant_id": "group-demo",
  "facility_id": "hospital-01",
  "patient_id": "pat-001",
  "encounter_id": "enc-001",
  "actor": "system",
  "payload": {
    "age": 67,
    "heart_rate": 118,
    "respiratory_rate": 28,
    "systolic_bp": 94,
    "temperature": 38.5,
    "spo2": 91,
    "wbc": 16.2,
    "creatinine": 1.8,
    "lactate": 2.7,
    "prior_admissions_30d": 1
  }
}
```

`payload` 的字段允许逐步补齐；评分前会按 `occurred_at` 合并该就诊截至评分时点的最近有效值，不会读取未来事件。响应和评分审计会显式返回 `missing_features`，便于人工判断数据完整性。重复且内容一致的 `event_id` 会被幂等忽略；同一个 `event_id` 携带不同就诊或内容会返回冲突。未知就诊、患者与院区不匹配或超范围数值会拒绝。生产接入必须补充单位、参考范围、采集设备和数据质量状态。

生产事件还应携带 `source_system`、`source_record_id`、`unit` 和 `quality_status`，用于数据血缘、去重、单位换算和质量门控；当前 API 已保留这些字段。

## 出院快照（P8）

`event_type == "discharge"` 的事件 `payload` 必须携带出院快照字段，缺任一字段或值非法会被
API（422）与导入脚本同时拒绝，错误信息给出字段名：

| 字段 | 类型 | 范围/约束 | 含义 |
| --- | --- | --- | --- |
| `discharge_destination` | str | `home` / `home_health` / `snf` / `other` | 出院去向 |
| `los_days` | float | 0–365 | 住院天数 |
| `diagnosis_codes` | list[str] | ≤50 条，每条 ≤16 字符 | 出院诊断编码 |
| `comorbidity_index` | float | 0–40 | 合并症指数 |
| `prior_utilization_12m` | int | 0–100 | 近 12 个月医疗利用次数 |
| `ed_visits_6m` | int | 0–100 | 近 6 个月急诊次数（LACE 的 E 维度） |

`readmission_30d` 只在出院事件上触发评分（见"风险分数"一节），特征即本快照的编码
（`app/features.py::encode_readmission_payload`）。

事件时间最多允许比服务端当前时间提前 `MAX_EVENT_FUTURE_SKEW_MINUTES` 分钟（默认 15 分钟），用于容忍院内时钟偏差并避免明显的未来数据进入实时风险队列。

## 风险分数

风险响应包含 `target_id`、`model_version`、`score`、`band`、`top_factors`、`missing_features`、`scored_at`、`status` 和 `human_review`。`target_id` 为 `deterioration_24h` 或 `readmission_30d`；`band` 为 `low`、`medium`、`high`、`critical`；处置状态为 `new`、`acknowledged`、`in_progress`、`resolved`。评分按目标触发：`deterioration_24h` 逐事件评分并始终出现在 `scores` 映射中；`readmission_30d` 只在 `event_type == "discharge"` 的出院事件上评分（单次评分，`as_of` 钉在首个出院事件时刻，后续生命体征不刷新），非出院事件的 `scores` 只含 deterioration。`score` 字段保留为 24 小时恶化目标的兼容字段。

## 结局表型（P5 标签派生）

- `deterioration_24h`：index 时刻起 24 小时窗口 `[index, index+24h)` 内发生 RRT（肾脏替代治疗）、
  非计划 ICU 转入、心跳骤停、死亡任一事件记为阳性；恰在 `index+24h` 整点的结局记为阴性（窗口上界严格小于）。
- `readmission_30d`：出院后 30 天内非计划再入院记为阳性（horizon_hours=720，index=出院时刻）。
- 结局事件行格式：`{encounter_id, occurred_at(ISO8601, naive 视为 UTC), outcome_type}`；
  `outcome_type` 白名单见 `scripts/derive_labels.py::OUTCOME_TYPES`。
- 演示数据：高风险患者集合 {2,5,9,14} 在首事件后 48h 发生 RRT；index 网格 1..96h，
  阳性率 96/1728 ≈ 5.6%（落在 5–15% 验收区间）。

## 风险工件 v2（P5）

- `feature_order: [str]`（可选）：推理遍历顺序；存在时必须与 `feature_names` 完全一致，否则启动拒绝。
- `preprocess: {feature: {median, mean, std}}`（可选，必须伴随 `feature_order`）：
  打分时标准化 `z = (value - mean) / std`（std≤0 视为 1），缺失特征用 `median` 填补并同时列入 `missing_features`。
- 无 `feature_order`/`preprocess` 的旧工件行为不变（硬编码中心化变换路径）。

## NEWS2 基线（P5）

- `scripts/baselines.py::news2_score`：RCP NEWS2 Scale 1 五组件（RR/SpO2/SBP/HR/体温），
  总分 0–15；任一单参数 3 分或总分 ≥7 → high，5–6 → medium，0–4 → low。
- 意识项（ACVPU）演示数据缺测，不计入总分——解读基线时须注明该局限。
- `news2_score` 同时作为模型输入（工件 `feature_order` 第 11 项）；评估报告输出
  `model`/`news2`/`model_with_news2` 三列与 AUPRC 增益，增益 ≤0.001 建议撤下
  （结论记录在 `news2_feature_gain.recommendation`）。
