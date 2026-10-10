# 项目检查与就绪度

## 当前结论

当前代码是一个可运行、可测试的医疗机器学习生产前验证切片。它已经覆盖“临床事件接入 → 特征聚合 → 风险评分 → 队列查询 → 人工复核 → 审计 → worker 重算”的闭环，适合方案评审、接口联调、演示和生产前验证。

它还不能直接作为医院临床诊断或治疗软件上线。原因不是 API 缺失，而是临床模型、身份体系、数据治理、验证证据和运营控制尚未达到真实医疗部署的要求。

## 覆盖矩阵

| 能力 | 当前状态 | 证据或缺口 |
| --- | --- | --- |
| 患者与就诊范围 | 已覆盖 | 租户、院区、患者和就诊一致性校验；查询按范围过滤 |
| 临床事件接入 | 已覆盖（演示契约） | Pydantic 校验、数值范围、有限性、UTC 时间、未知就诊拒绝 |
| 事件幂等 | 已覆盖 | 一致重复事件返回 `duplicate=true`；同 ID 不同内容返回 409 |
| 特征聚合 | 已覆盖 | 按事件时间合并最近有效值；支持 `as_of`，不读取未来事件 |
| 风险评分 | 已覆盖（演示模型） | 可解释逻辑回归、风险等级、主要因素、缺失特征 |
| 人工复核闭环 | 已覆盖 | 确认、处置、解决、备注和重算后状态保留 |
| 审计 | 已覆盖（演示级） | 事件评分和人工复核记录，按租户/院区读取 |
| 多进程本地存储 | 已覆盖（开发级） | SQLite WAL、事务、进程内锁；生产应换 PostgreSQL |
| API/worker 生命周期 | 已覆盖 | 应用关闭时释放连接；worker 显式注入 Store/Engine |
| 前端工作台 | 已覆盖（轻量演示） | 风险队列、详情、因素和人工状态；动态内容转义（`tests/test_frontend_logic.py` 断言 `safe()` 对 `& < > " '` 全量转义，`node --check` 守卫全文件语法） |
| 认证与授权 | 已具备生产适配器 | 开发环境 Bearer 演示令牌；生产通过 OIDC/JWT + JWKS 验签、issuer/audience、角色/租户/院区声明映射 |
| 消息总线 | 已具备 Redis Streams 适配 | `EVENT_BACKEND=redis` 时 API 先落库再入流；worker 使用 consumer group、ACK 和 pending reclaim；医院仍需高可用 Redis/院内消息系统、死信和容量演练 |
| 生产数据库 | 已具备适配器 | `PostgresStore`、复合租户键、JSONB、连接池和启动建表；仍需医院托管 PostgreSQL 的备份、灾备和迁移审批 |
| 临床目标拆分 | 已覆盖（平台层） | 两个 `target_id`、独立模型工件、阈值、分数、复核状态和模型卡；临床标签仍需医院确认 |
| 真实模型验证 | 已具备验证工具 | `scripts/evaluate_model.py` 支持时间切分、AUROC/AUPRC/Brier、灵敏度/特异度和亚组报告；真实数据和临床阈值仍需提供 |
| 模型注册与发布 | 已具备文件注册工作流 | `scripts/manage_model.py` 支持 promote、SHA-256 manifest 和 rollback；promote 门禁 = bootstrap CI 下界 ≥ 基线（deterioration→NEWS2、readmission→LACE）+ ECE<0.05 + 亚组灵敏度差 ≤10%（age 可用时）+ 逐院区 AUPRC ≥ 基线（≥2 院区时）+ 决策曲线平均增益 ≥0，不可用规则跳过并在 eval.json 的 `rules` 注明；生产仍需受控 Registry、审批、灰度和发布审计 |
| 可观测性 | 已具备基础指标 | `/api/metrics` 提供事件、评分、重复、队列、复核和错误计数；生产仍需 Prometheus 抓取、延迟/缺失率/漂移/校准告警、SLO 和告警系统 |
| 医疗合规 | 未覆盖（外部必需） | 需院内数据治理、脱敏、血缘、等保/安全控制映射、风险管理和合规评估 |

模型完整性和故障边界：开启 `MODEL_SIGNATURE_REQUIRED=true` 后，两个模型必须在 SHA-256
manifest 中有匹配摘要，否则应用拒绝启动。`MODEL_FALLBACK_ENABLED` 默认关闭；只有明确打开时，
模型不可用才会返回标记为 `model_unavailable` 的零分并把健康状态置为 `degraded`，该分数不能用于
临床告警。生产环境应优先 fail-closed，并由人工/规则流程接管。

## 机器学习使用边界

机器学习适合用于风险排序、早期预警、资源分配和出院管理优先级。事件格式校验、范围校验、权限、幂等、审计和最终临床处置使用确定性规则与人工流程更可靠；当前架构保留了这个边界。模型分数不能单独改变诊断、用药、手术或出院决定。

## 本次验证

在工作区使用以下检查通过：

```text
28 passed
ruff check app scripts tests
ruff format --check app scripts tests
python -m compileall -q app scripts tests
pip check: No broken requirements found
POSTGRES_PASSWORD=ci docker compose config --quiet
```

## P0 使用方式

### 双目标模型

```powershell
& .\.venv\Scripts\python.exe scripts\train_model.py --output data\model.json --target-id deterioration_24h --version approved-deterioration-v1
& .\.venv\Scripts\python.exe scripts\train_model.py --output data\readmission_model.json --target-id readmission_30d --version approved-readmission-v1
```

事件接入按目标触发返回分数：`deterioration_24h` 逐事件返回；`readmission_30d` 仅在 `event_type == "discharge"` 的出院事件上返回（单次评分，`as_of` 钉在首个出院事件时刻；非出院事件的 `scores` 只含 deterioration）。患者列表可以用 `?target_id=readmission_30d` 切换目标，人工复核请求也携带 `target_id`。

### 真实数据导入和验证

导入默认是 dry-run，确认摘要后再加 `--commit`：

```powershell
& .\.venv\Scripts\python.exe scripts\import_clinical_data.py --patients patients.json --events events.ndjson
& .\.venv\Scripts\python.exe scripts\import_clinical_data.py --patients patients.json --events events.ndjson --commit
& .\.venv\Scripts\python.exe scripts\evaluate_model.py --input labeled.csv --target-id deterioration_24h --group-column facility_id --split-time 2026-01-01T00:00:00+00:00
```

### 生产配置

将 `STORAGE_BACKEND=postgres`、`DATABASE_URL`、`OIDC_ENABLED=true`、`OIDC_ISSUER`、`OIDC_AUDIENCE` 和 `OIDC_JWKS_URL` 配置到密钥管理系统，并使用 `pip install -e ".[prod]"` 安装 PostgreSQL/OIDC 依赖。演示令牌必须保持关闭。

Docker daemon 在某些受限 Windows 环境不可访问时，镜像启动检查需要在具备 Docker Engine 权限的机器上执行；之前的镜像构建和 API/worker 健康检查已在可用环境完成。
