# 架构与演进

## 当前纵向切片

```text
院内事件/批数据
       |
       v
规范化事件契约 -> 特征快照 -> 风险引擎 -> 风险分层
       |                              |
       v                              v
数据/审计存储                 风险工作台 -> 人工复核/处置回流
```

本地演示用 SQLite 存储和静态工作台，启用 WAL、事务和进程内锁以支持 API 与 worker 多进程并发。生产配置可切换到带连接池和 JSONB 的 `PostgresStore`。患者和分数使用租户、就诊和目标复合键，事件时间统一为 UTC，并支持按时间点聚合，避免跨租户覆盖和未来数据泄漏。设置 `EVENT_BACKEND=redis` 后，事件会通过 Redis Streams consumer group 异步评分，失败消息支持 pending reclaim 和死信流。模型文件可由 `scripts/manage_model.py` 维护签名清单；医院可将该流程接入 MLflow Registry。API、事件契约和审计字段保持稳定。

## 服务边界

- `api`：鉴权、数据契约、查询、人工复核和审计。
- `risk engine`：特征计算、模型加载、评分、阈值和解释。
- `worker`：批量重算、失败重试和模型版本切换。
- `store`：患者/就诊状态、风险分数、处置状态和审计记录。
- `workbench`：按科室和风险等级查看患者，展示触发因素和处置状态。
- `security`：开发环境只接受显式 Bearer 演示令牌；生产环境可通过 OIDC/JWT + JWKS 验签，并由身份声明提供角色、租户和院区范围。

## 企业化下一步

1. 以患者主索引、就诊主索引和数据血缘替换演示存储。
2. 为每个院区增加 `tenant_id`、数据域权限和脱敏策略。
3. 使用时间切分和外部医院验证集管理模型晋级；模型卡与审批记录绑定版本。
4. 加入漂移、延迟、缺失率、校准度、告警负荷和处置闭环监控（P7 已落地 PSI/占比指标与再校准，见下节）。
5. 通过灰度院区上线；模型不可用时默认 fail-closed，由人工/规则流程接管。只有完成验证和审批后，才可启用显式的降级策略。

## 漂移监控与再校准（P7）

- **参考分布随工件**：训练时每个特征的 `drift_reference`（lo/hi + 10 等宽箱占比 + 缺失率）写入工件；再训练即刷新。
- **批量漂移检查**：`python scripts/monitor_drift.py --artifact <工件> --input <批次CSV>` 逐特征算 PSI（0.005 平滑、缺失率差并入），任一 PSI > 0.2 → 报告 `alert: true` 并 `exit 1`，对接方按退出码接告警系统（生产建议按月随再校准一起跑）。
- **漂移输入 CSV 的 schema 要求**（`monitor_drift`）：输入须包含工件 `feature_order` 的全部特征列——deterioration 为 21 维 v2（10 原始体征 + 10 个 `__missing` 指示列 + `news2_score` 派生列）；readmission 为 6 维出院快照编码（`diagnosis_code_count`、`discharge_destination_code` 两个派生列缺失时按出院源字段 `discharge_destination/los_days/diagnosis_codes/comorbidity_index/prior_utilization_12m/ed_visits_6m` 即时编码，R-RS-15）；其他缺失列按缺失入 bucket 计入 missing rate，不报错。
- **月度再校准**：`python scripts/recalibrate.py --artifact <工件> --input <标注CSV> --target-id <目标> [--gate-input <独立留出集>]` 只重拟合 `calibration` 块；门禁（CI 下界 ≥ 基线 + ECE + 三条规则）在临时候选工件上先跑，不过则原工件分毫不动、`exit 1`。同输入校准+门禁存在 in-sample 乐观性，真实数据务必提供独立 `--gate-input`。
- **校准方法与分辨率回退（Ruling R-P7-5/5a）**：校准默认 isotonic 回归；当拟合段唯一 y 水平 < 3（分数会坍缩成二值、摧毁阈值阶梯与分带）时自动回退 Platt，并在工件 `calibration` 块记录 `fallback_reason: isotonic_resolution`。demo deterioration 触发该回退属预期；真实数据阳性数充足时走真 isotonic。
- **在线指标**：`GET /api/metrics` 暴露近期分数（≤500 条）的 `medintel_band_share{band=...}` 与 `medintel_missing_rate{scope=...}` gauge（当前 SQLite 聚合，Postgres 适配器待补）。
