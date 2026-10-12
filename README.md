# MedIntel 医疗智能决策与质量中枢

这是一个面向医院集团的企业级机器学习项目骨架。首个纵向切片围绕成人普通住院患者，预测：

- 未来 24 小时临床恶化风险；
- 出院后 30 天再入院风险。

项目把机器学习放在有历史信号且可以闭环干预的环节，把规则、人工复核和运营约束保留在决策链上。当前版本可以用合成数据直接启动，提供事件触发评分、风险工作台、审计记录、模型解释和批处理 worker。

## 快速启动

需要 Python 3.11+。推荐使用 Docker（postgres 服务口令为必填项，内联提供或写入 `.env` 的 `POSTGRES_PASSWORD=`，部署时必须改为强口令）：

```powershell
POSTGRES_PASSWORD=local docker compose up --build
```

打开 <http://localhost:8000> 查看风险工作台，或打开 <http://localhost:8000/docs> 使用 API。

常用接口包括 `GET /api/patients`、`GET /api/patients/{encounter_id}`、`POST /api/events`、`POST /api/patients/{encounter_id}/review` 和 `GET /api/audit`。接口通过 `Authorization: Bearer <token>` 演示 RBAC；生产环境应替换为院内统一身份认证和短期令牌。

不使用 Docker 时：

```powershell
& .\path\to\python.exe -m venv .venv
& .\.venv\Scripts\pip install -e ".[dev]"
& .\.venv\Scripts\python scripts\train_model.py --output data\model.json --target-id deterioration_24h
& .\.venv\Scripts\python scripts\train_model.py --output data\readmission_model.json --target-id readmission_30d
& .\.venv\Scripts\uvicorn app.main:app --reload
```

首次启动会在 `data/store.db` 中生成合成住院数据和演示风险分数。运行 `python scripts/generate_demo_data.py --reset` 可以重置演示数据。

> ⚠️ **安全警告**：以下演示令牌**仅在 `APP_ENV=dev` 下有效**。非 dev 环境若 `DEMO_AUTH_ENABLED=true`，应用启动时直接抛出 `RuntimeError` 拒绝启动（fail-closed）；生产必须接入 OIDC 并设置 `DEMO_AUTH_ENABLED=false`。默认 compose 已把端口绑定到 `127.0.0.1`，仅限本机访问。

演示登录令牌：`demo-clinician-token`（临床查看和复核）、`demo-quality-token`（质量与模型卡）、`demo-admin-token`（跨院区管理）。请求使用 `Authorization: Bearer <token>`；这些令牌只在 `APP_ENV=dev` 下有效。

验证代码：

```powershell
& .\.venv\Scripts\pytest
```

训练脚本使用 scikit-learn 逻辑回归管线（中位数填补 → 标准化 → L2、`class_weight=balanced`，C∈{0.01,0.1,1} 按时间交叉验证 AUPRC 选取）生成两个独立目标的工件（`data/model.json` 和 `data/readmission_model.json`）；训练数据与演示/导入同源（合成患者 → 标签派生管道）。scikit-learn 仅是训练/评估依赖（`pip install -e ".[dev]"` 或 `".[train]"`），serving 保持纯标准库。真实数据导入和指标验证命令见 `docs/readiness-review.md`。

真实规模演练（2–3 万出院/入院合成队列）：一键跑通「生成 → 训练 → 患者级时间切分验证（整患者不跨折）→ 评估 → promote 门禁 → 模型报告卡 → PSI+KS 漂移自检」，按组计时预算并产出机读报告：

```powershell
& .\.venv\Scripts\python scripts\scale_rehearsal.py --scale 25000
# 产物：data/scale/（队列 CSV、工件、报告卡）与 data/rehearsal_report.{json,md}
# 单步也可用 --scale 调 generate_scale_data.py / train_model.py --input / validate.py --split patient / report_card.py
```

准备面试时从 [`docs/interview-guide.md`](docs/interview-guide.md) 开始。它包含 30 秒/3 分钟项目介绍、完整请求链路、模型指标、Redis 与一致性、医疗安全边界、20 个高频追问和七天训练计划。

生产异步事件：设置 `EVENT_BACKEND=redis`、`REDIS_URL` 和 `REDIS_STREAM` 后，API 会先持久化事件再写入 Redis Stream，worker 使用 consumer group 按触发策略计算（`deterioration_24h` 逐事件、`readmission_30d` 仅出院事件）并 ACK；处理失败的消息会留在 pending 列表中，下一次 worker 会尝试 reclaim，超过 `REDIS_MAX_DELIVERY_ATTEMPTS` 后转入死信流。生产应为 Redis 配置高可用、持久化、容量和死信运维策略。

本地启动 Redis/PostgreSQL 组合时使用 `POSTGRES_PASSWORD=local docker compose --profile future-infra up --build`，并在环境中设置 `STORAGE_BACKEND=postgres` 或 `EVENT_BACKEND=redis`。

模型完整性：`scripts/manage_model.py promote` 会把模型按 target/version 保存并生成 `manifest.json`。签名要求默认按环境判定：`MODEL_SIGNATURE_REQUIRED` 留空时 dev 关闭、非 dev 强制开启（fail-closed），启动日志会打印生效模式；无签名清单的 prod 演示部署必须显式 `MODEL_SIGNATURE_REQUIRED=false`。显式设置 `MODEL_SIGNATURE_REQUIRED=true` 和 `MODEL_SIGNATURE_PATH` 后，服务启动时校验 SHA-256。模型加载失败默认拒绝启动；只有显式启用 `MODEL_FALLBACK_ENABLED=true` 才会进入 `degraded` 健康状态并返回不可行动的零分。

模型加载失败行为矩阵：

| 场景 | 非 dev（prod/测试） | dev |
| --- | --- | --- |
| 工件缺失 | 拒绝启动 | 警告并使用默认模型（`/api/health` 的 `degraded_reasons` 给出类别码） |
| 验签失败 / 无签名清单 | 拒绝启动（默认强制验签） | 警告并回退默认模型 |
| 显式 `MODEL_FALLBACK_ENABLED=true` | 进入 `degraded`，返回不可行动的零分 | 同左 |

指标：`GET /metrics` 输出 Prometheus 文本格式——计数器带 `# TYPE <family> counter` 指标族声明与标签样本（如 `medintel_http_responses_total{code="200"} 1`），延迟指标为 `summary` 族（`_sum`/`_count`）。

## 目录

```text
app/
  api.py             FastAPI 应用和 HTTP 契约
  config.py          环境配置
  degrade.py         降级零分回退
  domain.py          临床事件、风险分数、审计事件和模型卡
  features.py        训练/serving 共享特征顺序与 landmark 特征
  main.py            应用装配、生命周期与中间件
  messaging.py       Redis Streams 事件发布与消费适配器
  observability.py   Prometheus 文本指标
  postgres_store.py  PostgreSQL 适配器
  risk.py            可解释的逻辑回归风险引擎
  security.py        演示版 RBAC 和操作审计
  storage.py         存储后端选择
  store.py           SQLite 开发实现
  synthetic.py       合成数据与事件生成器
  validation.py      事件时间与导入校验
  worker.py          批量重评分 worker
  static/
    index.html       轻量风险工作台（壳页）
    app.js           工作台前端逻辑
scripts/
  baselines.py       NEWS2 与 LACE 基线评分
  build_features.py  landmark 七元组特征构建
  derive_labels.py   标签派生（demo CLI）
  evaluate_model.py  模型评估与三列对比（--split patient|encounter）
  generate_demo_data.py
  generate_scale_data.py  2–3 万规模合成队列生成（G1）
  import_clinical_data.py  真实数据导入与单事务校验
  manage_model.py    模型 promote、SHA-256 清单与 rollback
  monitor_drift.py   PSI + KS 双指标漂移监控（派生特征即时编码）
  report_card.py     自动模型报告卡（10 章节，md + json）
  scale_rehearsal.py 端到端规模演练编排（分组时间预算）
  train_model.py     scikit-learn 逻辑回归训练脚本（--input 规模 CSV）
  validate.py        滚动时间切分（患者级防泄漏）与 bootstrap CI
docs/
  architecture.md    企业架构和演进路线
  data-contract.md   事件与风险接口契约（含结局表型/工件 v2/NEWS2）
  model-card.md      首期模型卡和临床使用边界
  readiness-review.md 项目检查、覆盖矩阵和上线前缺口
  clinical-pilot-protocol.md 临床标签、静默运行和上线验收模板
  interview-guide.md 面试准备指南
tests/
```

## 重要边界

当前项目是研究和生产前验证骨架，不是医疗器械软件或临床诊断产品。上线前必须完成真实院内数据治理、独立时间切分验证、亚组公平性分析、临床前瞻性评估、风险管理、网络安全和人工复核流程。模型分数只用于分层和提醒，不能单独改变诊疗方案。
