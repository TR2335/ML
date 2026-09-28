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

训练脚本使用纯 Python 梯度下降生成两个独立目标的演示模型（`data/model.json` 和 `data/readmission_model.json`），方便在没有 GPU 或科学计算环境的情况下演示完整的训练到服务流程；真实项目应把它替换为经过审批的训练流水线。真实数据导入和指标验证命令见 `docs/readiness-review.md`。

准备面试时从 [`docs/interview-guide.md`](docs/interview-guide.md) 开始。它包含 30 秒/3 分钟项目介绍、完整请求链路、模型指标、Redis 与一致性、医疗安全边界、20 个高频追问和七天训练计划。

生产异步事件：设置 `EVENT_BACKEND=redis`、`REDIS_URL` 和 `REDIS_STREAM` 后，API 会先持久化事件再写入 Redis Stream，worker 使用 consumer group 计算两个目标并 ACK；处理失败的消息会留在 pending 列表中，下一次 worker 会尝试 reclaim，超过 `REDIS_MAX_DELIVERY_ATTEMPTS` 后转入死信流。生产应为 Redis 配置高可用、持久化、容量和死信运维策略。

本地启动 Redis/PostgreSQL 组合时使用 `POSTGRES_PASSWORD=local docker compose --profile future-infra up --build`，并在环境中设置 `STORAGE_BACKEND=postgres` 或 `EVENT_BACKEND=redis`。

模型完整性：`scripts/manage_model.py promote` 会把模型按 target/version 保存并生成 `manifest.json`。签名要求默认按环境判定：`MODEL_SIGNATURE_REQUIRED` 留空时 dev 关闭、非 dev 强制开启（fail-closed），启动日志会打印生效模式；无签名清单的 prod 演示部署必须显式 `MODEL_SIGNATURE_REQUIRED=false`。显式设置 `MODEL_SIGNATURE_REQUIRED=true` 和 `MODEL_SIGNATURE_PATH` 后，服务启动时校验 SHA-256。模型加载失败默认拒绝启动；只有显式启用 `MODEL_FALLBACK_ENABLED=true` 才会进入 `degraded` 健康状态并返回不可行动的零分。

## 目录

```text
app/
  api.py             FastAPI 应用和 HTTP 契约
  config.py          环境配置
  domain.py          临床事件、风险分数、审计事件和模型卡
  risk.py            可解释的逻辑回归风险引擎
  store.py           SQLite 开发实现；生产替换为 PostgreSQL 适配器
  synthetic.py       合成数据与事件生成器
  security.py        演示版 RBAC 和操作审计
  worker.py          批量重评分 worker
  messaging.py       Redis Streams 事件发布与消费适配器
  observability.py   Prometheus 文本指标
  static/index.html  轻量风险工作台
scripts/
  train_model.py     纯 Python 梯度下降训练脚本
  manage_model.py    模型 promote、SHA-256 清单与 rollback
  generate_demo_data.py
  docs/
  architecture.md    企业架构和演进路线
  data-contract.md   事件与风险接口契约
  model-card.md      首期模型卡和临床使用边界
  readiness-review.md 项目检查、覆盖矩阵和上线前缺口
  clinical-pilot-protocol.md 临床标签、静默运行和上线验收模板
tests/
```

## 重要边界

当前项目是研究和生产前验证骨架，不是医疗器械软件或临床诊断产品。上线前必须完成真实院内数据治理、独立时间切分验证、亚组公平性分析、临床前瞻性评估、风险管理、网络安全和人工复核流程。模型分数只用于分层和提醒，不能单独改变诊疗方案。
