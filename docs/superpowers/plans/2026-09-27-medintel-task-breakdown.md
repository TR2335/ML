# MedIntel 项目任务拆解与实施计划（主文档）

> **For agentic workers:** 本文档是主任务拆解（WBS）。每个阶段（P0–P8）在执行前需按 superpowers:writing-plans 展开为独立的详细实施计划（含逐步 TDD 步骤与代码）。步骤使用 `- [ ]` 复选框语法跟踪。

**Goal:** 把审查发现的 66 个问题与双版本 ML 算法路线拆解为可执行、可验收、有依赖顺序的任务集。

**Architecture:** 三条执行线 + 一个自助数据接入——①缺陷修复（P0–P4，按危级排序）；②ML 数据管线与双版本模型（P5–P7）；③再入院目标改造（P8，独立子线）；④数据自助接入（P9，用户按 `docs/data-integration-design.md` 自行执行，不阻塞实现）。所有 ML 任务共用同一套标签/landmark/serving 工件契约，升级只换 estimator。

**Tech Stack:** Python 3.11+、FastAPI、SQLite/PostgreSQL、Redis Stream、scikit-learn、LightGBM（P7）、pytest、ruff。

**Spec:** 依据本会话三项产出：《全量审查报告》（66 问题）、《ML 可行性评估》、《双版本算法方案》。修复类任务以审查报告条目编号为追溯锚点。

---

## 整体项目蓝图

**定位**：事件驱动的住院风险评分系统——临床事件进入 API，风险引擎按模型工件打分，工作台呈现、审计全程留痕；ML 训练管线与在线服务解耦演进。**P0–P8 全部现在实现（用 demo/合成数据验证，不等真实数据）；真实数据接入（P9）由用户按 `docs/data-integration-design.md` 自助完成。**

### 系统全貌与数据流

```text
数据源（实现阶段：合成演示数据；真实数据自助接入见 P9 与设计文档）
   │  POST /api/events  │  scripts/import_clinical_data.py
   ▼
FastAPI 应用层 api.py + security.py(RBAC/审计) + config.py
   ├─ 持久化: store.py(SQLite) ⇄ postgres_store.py(PostgreSQL)
   └─ 事件: sync 直接评分 │ EVENT_BACKEND=redis → Redis Stream
   ▼
worker.py 评分/批量重评分 ──► risk.py 风险引擎打分
   ▲                            ▲
   │                            │ 加载工件 data/*.json
   └── scores 回写存储 ◄────────┘
             │
             ▼
工作台 static/index.html (GET /api/patients*) · observability 指标 · degrade 降级

模型工件生产线（与在线侧解耦）:
train_model.py 训练 → evaluate_model.py 评估(NEWS2 对比)
  → manage_model.py 门禁/promote → risk.py 加载打分
```

### 模块职责与计划演进

| 层 | 文件 | 职责 | 本计划中的演进 |
|---|---|---|---|
| API 与安全 | `api.py` `main.py` `security.py` | HTTP 契约、RBAC、审计 | P1 高危修复、T4.2 端点收敛 |
| 领域模型 | `domain.py` | 事件/分数/审计 schema | T1.2 `as_of`、T3.3、T4.5(7) |
| 存储 | `store.py` `postgres_store.py` `storage.py` | 双后端持久化 | T1.1/T1.2/T2.2 对同一函数连续三改，必须按序执行 |
| 事件与异步 | `worker.py` `messaging.py` | 评分循环、Stream 消费 | T2.3 对账、T3.4 崩溃韧性 |
| 风险引擎 | `risk.py` | 可解释打分、工件加载 | T3.2 原子加载、T5.3 工件契约 v2、T7.2 树归因 |
| ML 管线 | `scripts/*`（含新建 derive_labels/build_features/baselines 等） | 训练→校准→评估→门禁 | P5 新建、P6 重写、P7 升级、P9 真实数据再训练 |
| 展示 | `static/index.html` | 风险工作台 | T1.4 XSS、T4.1 目标语义 |
| 可观测与配置 | `observability.py` `degrade.py` `config.py` | 指标、降级、fail-closed 配置 | T1.3、T3.5、T4.5(8) |

### 整体路线（三步走，全程不等真实数据）

1. **第一步 · 工程与产品完善（P0–P4）**——不依赖任何数据：先立工程基线，再按危级修完审查问题，安全/并发/文档/UI 全部到位（里程碑 M-A → M-B → M-C）。
2. **第二步 · ML 底座双版本闭环（P5–P6）**——用 demo 合成数据：把「标签 → 特征 → 训练 → 校准 → 评估（NEWS2 对比）→ promote 门禁 → 离在线一致」全链路跑通，工件契约在此定型（里程碑 M-D）。
3. **第三步 · 充足版与再入院（P7–P8）**——同样用合成数据实现与验证，不等真实数据：LightGBM 升级（P7）与再入院目标（P8）直接开发（里程碑 M-E → M-F）。

> P0–P8 的全部验收均不依赖真实数据；真实数据接入（P9）由用户按《`docs/data-integration-design.md`》自助执行（里程碑 M-G），不阻塞主线实现。

## Global Constraints

- 工件 JSON 形状保持向后兼容：`weights/bias/feature_names/thresholds/model_version` 字段不可删除，只可新增（`preprocess`、`calibration`、`trees`）。
- 所有修复必须带回归测试；测试不得使用 `.only`/`.skip` 占位。
- 现有 28 个测试全程保持通过（`python -m pytest tests/`，若 `.pytest-tmp` 权限未修可用 `--basetemp=<临时目录>` 绕过）。
- 每个任务独立提交，commit message 遵循仓库现有风格。
- 不引入重型 ML 运行时到 serving 依赖（serving 保持纯 Python 标准库 + pydantic）。
- 中文 UI 文案与 `docs/` 文档用语保持现有风格。

## Review Focus（审查者最应关注的五类失败）

1. **幂等语义回归**：任何改 `upsert_score` 的任务，必须同时验证"重打分保留人工复核"（tests/test_store.py:33）与"review 端点可多次流转"（T1.1 新测试）两条相反需求同时成立。
2. **认证 fail-closed**：T1.3 改默认配置后，dev 模式演示令牌仍须可用（本地联调不破坏），非 dev 必须拒绝演示令牌——两个方向都要有测试。
3. **时间语义一致性**：T1.2 改评分覆盖逻辑后，sync 与 redis 两后端对同一事件序列产出的分数必须一致（新增跨后端一致性测试）。
4. **训练/服务一致性**：P5–P6 的任何工件格式变更，必须验证 `evaluate_model.py` 与 `RiskEngine` 读取同一工件产出相同分数（由 T6.5 golden dataset 显式钉住该契约）。
5. **医疗 UI 语义正确**：T4.1 改前端后，切换目标时不得显示错误目标的分数（回归场景：目标 B 无分数而目标 A 有分数）。

---

## 阶段总览与依赖

```
P0 工程基线（半天）──────────┐
P1 高危修复（5任务）─────────┼─→ P2 中危·数据完整性 ─→ P4 中低危·API/前端/文档
                             └─→ P3 中危·模型与配置 ─┘
P5 ML 数据管线（与 P1–P4 并行，依赖 P0）─→ P6 不足版模型 ─→ P7 充足版（合成数据验证）
P8 再入院改造（独立，依赖 P5 的工件契约）
P9 数据自助接入（用户按 docs/data-integration-design.md 执行，不阻塞 P0–P8）
```

优先级：P1 > P0 > P2 > P3 > P5 > P4 > P6 > P8 > P7 > P9（P0–P8 现在全部实现，不等真实数据；P9 由用户自助执行）。

---

## P0 工程基线（先做，半天）

### T0.1 修复测试环境与 git 忽略
- **文件**: `pyproject.toml`、`.gitignore`、本地 `.pytest-tmp`
- **动作**: 删除权限损坏的 `.pytest-tmp*`/`.pytest_cache`（Windows ACL 修复；当前仓库残留为 `.pytest-tmp2/`，执行时以实际目录为准）；`addopts` 的 `--basetemp` 改为相对临时路径；`.gitignore` 增加 `.omc/`、`model-registry/`，并把现有 `.pytest-tmp/` 放宽为 `.pytest-tmp*/`。
- **验收**: `python -m pytest tests/ -q` 原样命令直接通过 28 项；`git status` 不再出现 `.omc/`。
- **依赖**: 无

### T0.2 清理 ruff 10 项
- **文件**: `app/api.py:43`、`app/messaging.py:89`、`app/{main,risk,store,worker}.py`、`tests/test_api.py`、`scripts/evaluate_model.py:14`、`scripts/{import_clinical_data,manage_model}.py`
- **动作**: `ruff check --fix` 处理 6 项 I001；两处 BLE001 裸捕获改为具体异常或加注释豁免（`# noqa: BLE001` 需附理由）；UP035 改 `collections.abc`；两处 TRY004 改 `TypeError`。
- **验收**: `ruff check app scripts tests` 退出码 0；`ruff format --check` 通过。
- **依赖**: 无

### T0.3 添加 .dockerignore
- **文件**: 创建 `.dockerignore`
- **内容**: `.venv`、`.git`、`__pycache__`、`data/`、`.omc`、`tests/`、`.pytest*`、`.ruff_cache`、`*.egg-info`、`docs/`
- **验收**: `docker build` 上下文不含 `.venv`（构建日志 size 显著下降）。
- **依赖**: 无

### T0.4 CI 补强
- **文件**: `.github/workflows/ci.yml`
- **动作**: Python 矩阵 `[3.11, 3.12]`；增加 `pip-audit` 步骤；增加 `docker compose config --quiet` 校验；加 pip cache。
- **验收**: CI 在两个 Python 版本全绿；故意引入已知 CVE 依赖时 pip-audit 失败。
- **依赖**: T0.2（否则 lint 步骤先挂）

---

## P1 高危修复（5 个任务，每个含回归测试）

### T1.1 人工复核状态流转失效（审查 H1）
- **文件**: `app/store.py`（`upsert_score`）、`app/postgres_store.py:310`、`app/api.py:270`
- **根因**: `upsert_score` 的"重打分保留复核"逻辑（store.py:421-425）把 review 端点写入的新状态还原。
- **修法**: `upsert_score(score, *, preserve_review: bool = True)`；review 端点改调 `store.upsert_score(score, preserve_review=False)`。Postgres 侧同样加参数并同步修改 SQL 路径。
- **测试**: 新增 `tests/test_api.py::test_review_status_transitions`——依次提交 `acknowledged → in_progress → resolved`，每次 GET 断言状态与 note 已更新；保留 `tests/test_store.py` 现有"重打分保留复核"测试（两条需求都过才算完）。
- **验收**: 新旧测试全绿。
- **依赖**: 无（最高优先）

### T1.2 评分覆盖的时间语义（审查 H3）
- **文件**: `app/store.py:413`（`upsert_score`）、`app/postgres_store.py:307`、`app/worker.py:47`
- **根因**: 单行 last-write-wins，旧事件重投递会用过期快照覆盖新评分；两后端 as_of 语义不一致。
- **修法**: `RiskScore` 增加 `as_of: datetime` 字段（存储 JSON 向后兼容：缺失时视为 `occurred_at`）；`upsert_score` 仅当 `score.as_of >= previous.as_of` 才覆盖分数数值字段（否则跳过写入，返回 `skipped_stale`）；**review 状态字段不受此判断影响**（与 T1.1 协同）。
- **测试**: `tests/test_store.py::test_stale_score_does_not_overwrite`（先写 E2 时刻分数，再写 E1 时刻，断言分数保持 E2）；`tests/test_worker.py` 增加"乱序事件最终分数基于最新 as_of"。
- **验收**: 两后端对同一事件序列（乱序投递 + 重投递）产出相同分数。
- **依赖**: T1.1（字段与合并语义需一起设计，避免两次改同一函数）

### T1.3 默认部署安全化（审查高危·默认配置）
- **文件**: `app/config.py:7,30-33`、`app/security.py:106-114`、`docker-compose.yml:5,19,27`、`README.md:34`、`app/static/index.html:33`
- **修法**:
  1. `validate_auth_configuration()`（security.py 启动校验）增加断言：`app_env != "dev"` 且 `demo_auth_enabled` → 直接 `RuntimeError`（fail-closed，非告警）。
  2. compose 默认 `APP_ENV:-prod`、`DEMO_AUTH_ENABLED:-false`、端口改 `"127.0.0.1:8000:8000"`。
  3. README 演示令牌段落加醒目警告框，注明仅 `APP_ENV=dev` 有效。
- **测试**: `tests/test_production_controls.py::test_non_dev_rejects_demo_auth`（构造 `Settings(app_env="prod", demo_auth_enabled=True)` 断言抛错）；`test_dev_demo_auth_still_works` 防回归。
- **验收**: prod 配置启动被拒；dev 本地联调行为不变。
- **依赖**: 无

### T1.4 前端 XSS 与安全响应头（审查前端 XSS）
- **文件**: `app/static/index.html:34`、`app/main.py:74-80`
- **修法**: `safe()` 改为显式转义 `& < > " '`（替换 textContent/innerHTML 技巧）；`create_app` 内加中间件为 `/` 页面响应注入 `Content-Security-Policy: default-src 'self'` 与 `X-Content-Type-Options: nosniff`。
- **测试**: `tests/test_api.py::test_index_security_headers`（GET `/` 断言 CSP 头存在）；对 `safe()` 逻辑的纯函数测试（注入 `x" onmouseover="alert(1)` 断言输出全被转义）。
- **验收**: 属性注入 payload 在渲染后不含可执行引号；测试全绿。
- **依赖**: 无

### T1.5 认证依赖不再阻塞事件循环（审查 H2）
- **文件**: `app/security.py:145`
- **修法**: `async def dependency` 改为普通 `def dependency`（FastAPI 自动放入线程池执行）；JWKS 获取保持同步（现已被隔离出事件循环）。
- **测试**: `tests/test_api.py::test_auth_dependency_runs_in_threadpool`（用 marker 断言依赖在非事件循环线程执行——`threading.get_ident()` 对比主线程 id）。
- **验收**: 测试通过；httpx 全套 API 测试无回归。
- **依赖**: 无

---

## P2 中危·数据完整性与并发

### T2.1 事件与审计 ID 碰撞（审查 M1）
- **文件**: `app/domain.py:43-45`、`app/store.py:245,256,440`、`app/postgres_store.py:136-148`
- **修法**: `event_id`/`audit_id` 改完整 `uuid4().hex`（32 字符）；唯一键语义保持现结构（存量 10 位 ID 无需回填，碰撞概率随新格式引入而衰减）。若保留短 ID 兼容：唯一键改 `(tenant_id, event_id)`。
- **测试**: `tests/test_store.py::test_cross_tenant_event_id_collision_accepted`（构造两租户相同 event_id，断言均入库且幂等各自成立）。
- **依赖**: 无

### T2.2 upsert_score 并发丢失复核（审查 M4）
- **文件**: `app/store.py:413`、`app/postgres_store.py:307`
- **修法**: SQLite——读改写已在 `self._lock, self.connection` 事务内，补 `BEGIN IMMEDIATE`；Postgres——`SELECT ... FOR UPDATE` 单事务内完成读与写。
- **测试**: 并发测试：两线程同时 upsert（一个改 status、一个改 score），断言最终两字段都保留（Postgres 路径暂标记为需要 `future-infra` profile 的集成测试）。
- **依赖**: T1.1、T1.2（同函数第三次改造，顺序执行避免冲突）

### T2.3 事件-评分对账补偿（审查 M5）
- **文件**: `app/worker.py`、`app/store.py`、`app/postgres_store.py`
- **修法**: `rescore_once()` 扩展：增加 `reconcile()`——查询 `scores` 中缺失或 `as_of < events.max(occurred_at)` 的 encounter 列表重算（两后端共用 SQL 语义）；redis 模式 worker 每 N 轮调用一次。
- **测试**: `tests/test_worker.py::test_reconcile_scores_missing_events`（直接 insert 事件不入队，调 reconcile，断言分数生成）。
- **依赖**: T1.2（as_of 比较是判定依据）

### T2.4 导入脚本契约对齐（审查导入校验缺口）
- **文件**: `scripts/import_clinical_data.py:51-59,67-81`
- **修法**: 复用 `api.py:125-129` 的未来偏移校验逻辑（抽成 `app/validation.py` 共用函数，两处调用）；导入包裹在单事务，失败时打印 partial summary（accepted/failed 计数 + 最后成功键）。
- **测试**: 新建 `tests/test_import.py`：未来时间戳 CSV 被拒；中途失败输出含计数的摘要。
- **依赖**: 无

---

## P3 中危·模型与配置

### T3.1 模型签名非 dev 默认开启（审查 M7）
- **文件**: `app/config.py:23`、`app/risk.py:163-164`、`docker-compose.yml`、`.env.example:17`
- **修法**: Settings 的 `model_signature_required` 默认改 `None: bool | None`；`_strict_model_mode()`（非 dev）下 `None` 视为 True（fail-closed），dev 下 `None` 视为 False；启动日志打印生效模式。
- **测试**: `tests/test_production_controls.py::test_prod_requires_signature_by_default`（无签名 manifest + prod → 启动抛 RuntimeError）；dev 默认仍可跑。
- **依赖**: 无

### T3.2 模型加载原子性与缺失日志（审查 M2/L3）
- **文件**: `app/risk.py:75-127`
- **修法**: `_load` 内全部校验通过后，用局部变量集一次性提交 `self.version/algorithm/label_definition/thresholds/weights/bias`（异常时状态零污染）；文件缺失分支（:75-83）补 `LOGGER.warning` 并写入 `degraded_reason` 供 `/api/health` 可见（不改分数行为，只加可观测性）。
- **测试**: `tests/test_production_controls.py::test_invalid_thresholds_leaves_version_unchanged`（阈值非法 + dev → 断言 `engine.version` 仍为配置默认值而非制品值）。
- **依赖**: 无

### T3.3 payload 布尔值返回 422（审查 M3）
- **文件**: `app/domain.py:74-75`
- **修法**: validator 内 `raise TypeError` → `raise ValueError`。
- **测试**: `tests/test_api.py::test_boolean_payload_value_returns_422`（提交 `{"payload":{"age":true}}` 断言 422 非 500）。
- **依赖**: 无（一行修复，可并入任何批次）

### T3.4 worker 崩溃韧性与编排（审查 M10）
- **文件**: `app/worker.py:125-127`、`docker-compose.yml`
- **修法**: sync 主循环包 `try/except Exception` + `LOGGER.exception` + 指数退避 sleep（对齐 redis 分支 worker.py:101-115 的处理）；compose 全服务加 `restart: unless-stopped`，worker 加 healthcheck（基于时间戳文件或简单 HTTP 探活）。
- **测试**: `tests/test_worker.py::test_sync_loop_survives_rescore_failure`（monkeypatch `rescore_once` 抛错一次，断言循环继续且记录日志）。
- **依赖**: 无

### T3.5 配置安全加固（审查数据库默认口令/L16）
- **文件**: `app/config.py:13,42`、`docker-compose.yml`
- **修法**: 非 dev 且 `storage_backend=postgres` 且 `DATABASE_URL` 为默认值 → 启动拒绝（复用 T1.3 的启动校验框架）；compose postgres 改 `POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?required}`；`extra="ignore"` 在非 dev 改 `extra="forbid"`（拼错环境变量直接报错）。
- **测试**: `tests/test_production_controls.py::test_prod_rejects_default_database_url`、`test_prod_rejects_unknown_setting`。
- **依赖**: T1.3（复用其启动校验框架）

### T3.6 审计机构语义（审查 M8）
- **文件**: `app/security.py:169`、`app/store.py:456`、`app/postgres_store.py:351`
- **修法**: audit 行拆 `actor_facility_id` 与 `resource_facility_id` 双字段（迁移：SQLite 加列默认回填现值）；`/api/audit` 过滤按 resource 字段；兼容期 actor 字段同时保留。
- **测试**: `tests/test_api.py::test_admin_action_visible_in_facility_audit`（admin 操作 hospital-01 资源 → 以 hospital-01 过滤能查到）。
- **依赖**: 无

---

## P4 中低危·API / 前端 / 文档

### T4.1 前端目标分数回退与静默失败
- **文件**: `app/static/index.html:35-38`
- **修法**: 删除 `|| p.risk` 回退，目标无分数时显示"暂无该目标的风险分数"（复用现有 empty 分支文案）；所有 `fetch` 包 `.catch` 与 `response.ok` 检查，失败时 UI 显示错误提示而非静默停留旧数据。
- **测试**: 以纯函数方式拆出 `selectScore(p, target)`，`tests/test_frontend_logic.py` 断言回退行为已移除（目标 B 无分数 → 返回 null）。
- **依赖**: 无

### T4.2 端点信息收敛
- **文件**: `app/api.py:36-64,149-151,173-196`、`app/main.py:49`
- **修法**: health 的 `degraded_reasons` 对外只给错误类别码，绝对路径仅进日志；非 dev 关闭 `/docs`（`docs_url=None`）；幂等比对纳入 `occurred_at`；sync 重复路径补 `medintel_events_scored_total` 指标与 audit。
- **测试**: `tests/test_api.py::test_health_does_not_leak_paths`、`test_duplicate_event_with_different_time_rejected_409`。
- **依赖**: T1.3（非 dev 行为改动集中一处）

### T4.3 文档一致性修复
- **文件**: `README.md:50,54-78`、`docs/readiness-review.md:22`、`.env.example:37`、`docs/architecture.md`、`docs/data-contract.md`
- **修法**: ①交换 `architecture.md` 与 `data-contract.md` 的内容（当前两文件内容与文件名相反：architecture.md 装的是《数据契约》，data-contract.md 装的是《架构与演进》），使文件名与内容对应——此项必须先完成，否则 T5.1/T5.3/T8.1 写入 `data-contract.md` 的结局表型与工件契约会落进错误文件；②README 模型失败行为补环境矩阵（非 dev fail-closed / dev 警告+回退 / fallback 显式降级）；③目录树修正（docs 平级、补 main.py、storage.py、postgres_store.py、degrade.py 及 scripts 下 evaluate/import 两脚本）；④readiness-review 的"动态内容转义"表述补 `safe()` 缺口已修的证据（T1.4 完成后）；⑤删除死配置 `MLFLOW_TRACKING_URI` 或注明占位。
- **验收**: `head -1 docs/architecture.md` 输出 `# 架构与演进`、`head -1 docs/data-contract.md` 输出 `# 数据契约`；README 目录树与 `ls app/ scripts/` 逐一对应；其余文档逐条与代码核对（以本计划各任务内嵌的问题描述为追溯锚点）。
- **依赖**: T1.4、T4.2（先有行为后写文档）；文件内容交换本身无依赖，可最先做

### T4.4 list_patients 分页
- **文件**: `app/api.py`、`app/store.py:292-307`、`app/postgres_store.py:176-198`
- **修法**: 增加 `limit/offset` 查询参数（默认 limit=200、上限 1000），SQL 加 `LIMIT ? OFFSET ?`，响应保留 `X-Total-Count` 头；前端 `load()` 传 limit。
- **测试**: `tests/test_api.py::test_patient_pagination`。
- **依赖**: 无

### T4.5 低危批量清理
- **文件/动作**（一批提交，逐条测试）:
  1. `security.py:129` Bearer 前缀大小写不敏感解析 → 测试 `bearer <token>` 也通过；
  2. `security.py` JWT 校验加 `leeway=30` → 测试 25 秒时钟偏移仍通过；
  3. `store.py:23` `read_bytes()[:1]` → `open().read(1)` → 测试构造时不整读文件（mock 断言 read 参数）；
  4. 删除 `event_exists` 死代码（`store.py:268`）→ 全仓 grep 无引用；
  5. `postgres_store.py` 补 schema 迁移（对齐 SQLite `_migrate_*`）→ 集成测试挂 `future-infra`；
  6. `worker.py:120` event_backend 拼写错误时 worker 与 API 同样抛 ValueError → 测试非法值两侧一致；
  7. `domain.py:55` payload 值长度上限（单值 max 4KB、事件总 payload 64KB）→ 测试超限 422；
  8. `observability.py` 补 `# TYPE` 指标族 + 状态码改 label → 文档注明 Prometheus 抓取格式。
- **依赖**: T2.1（同文件改动批次协调）

### T4.6 端到端集成测试（兑现 Review Focus #3 的跨后端一致性载体）
- **文件**: 新建 `tests/integration/test_backends_consistency.py`、`tests/integration/test_redis_failure.py`、`pyproject.toml`（注册 `integration` marker）
- **背景**: `readiness-review.md` 承认"真实 PostgreSQL/Redis/OIDC 仍需在集成环境做容器和故障注入测试"；T1.2 验收要求"两后端对同一事件序列产出相同分数"，但没有可执行的测试载体——本任务补上这个载体。
- **修法**:
  1. `pyproject.toml` 注册 `markers = ["integration: requires docker compose --profile future-infra"]`，集成测试默认被 `-m "not integration"` 排除，本地无 Docker 不阻塞 28 项单元测试。
  2. `test_backends_consistency.py`：用 `STORAGE_BACKEND=postgres` 与 `STORAGE_BACKEND=sqlite` 各起一个 app 实例，按同一顺序提交事件（含乱序投递与同 event_id 重投递），断言两后端 `GET /patients/{id}` 返回的 `score/band/as_of` 逐字段相等（浮点差 < 1e-9）。
  3. `test_redis_failure.py`：`EVENT_BACKEND=redis` 下先正常入库再 kill Redis 容器，断言 API 对后续事件返回 503 且审计含 `event_publish_failed`；恢复容器后用相同 `event_id` 重试，断言事件最终评分成功（幂等重发路径）。
  4. CI 增加 `integration` job：`docker compose --profile future-infra up -d` → 等待健康 → `pytest -m integration` → `docker compose --profile future-infra down -v`。
- **测试**: 本任务的测试即上述两个文件；断言目标就是 Review Focus #3（sync 与 redis 后端对同一事件序列产出相同分数）与 README 声称的"发布失败返回 503、调用方可重试"。
- **验收**: 有 Docker Engine 的机器上 `pytest -m integration` 全绿；`pytest -m "not integration"` 仍为单元测试集合且全绿。
- **依赖**: T1.2、T2.2（as_of 与并发语义定型后才能断言一致）、T3.4（worker 退避行为）

---

## P5 ML 数据管线（双版本共用底座，与 P1–P4 并行）

### T5.1 标签派生 `scripts/derive_labels.py`
- **文件**: 创建 `scripts/derive_labels.py`；`app/synthetic.py` 扩展
- **接口**:
  - Produces: `derive_labels(events, outcomes, index_times, horizon_hours=24) -> list[LabeledLandmark]`；CLI 输出 CSV 列 `encounter_id, index_time, label` + 特征列
  - 结局表型定义写入 `docs/data-contract.md`：`deterioration_24h` = index 后 24h 内 {RRT, 非计划 ICU 转入, 心跳骤停, 死亡} 任一；`readmission_30d` = 出院后 30 天非计划再入院。
  - `synthetic.py` 增加 `demo_patients_with_outcomes()`：按现有高风险患者集合（index {2,5,9,14}）生成 RRT 结局，使演示管道与真实管道同构。
- **测试**: 新建 `tests/test_labels.py`——窗口边界（严格 `< 24h` 算阳性，24h 整点算阴性，显式定死）、窗口外结局不计入、无结局为 0。
- **验收**: `python scripts/derive_labels.py --demo` 产出带 label 的 CSV，阳性率落在 5–15%。
- **依赖**: T0.1

### T5.2 landmark 特征构建 `scripts/build_features.py`
- **文件**: 创建 `scripts/build_features.py`；`app/features.py`（共享特征定义）
- **接口**:
  - Produces: `build_landmarks(patient_events, index_times, window_hours=24) -> list[dict]`；每特征产出 `最新值/均值/斜率/最小/最大/缺失指示/距上次测量` 七元组（现有 10 特征 → 约 70 维）
  - 斜率 = 窗口内对时间的最小二乘系数（样本 <3 时置 None）
  - 特征顺序常量 `FEATURE_ORDER` 写入 `app/features.py`，训练与 serving 共同 import（消灭 risk.py 硬编码变换的前置）
- **测试**: 新建 `tests/test_features.py`——斜率计算正确性（构造线性上升序列）、缺失指示、窗口严格早于 index time（防泄漏）。
- **依赖**: T5.1

### T5.3 工件契约 v2
- **文件**: `app/risk.py:84-118`、`scripts/train_model.py`、`docs/data-contract.md`
- **修法**: 工件 JSON 新增可选 `preprocess` 块（`{feature: {median, mean, std}}`）与 `feature_order`；`_score` 优先用工件内参数，缺失时回退现有硬编码变换（向后兼容）；`feature_names` 校验改为与 `feature_order` 一致。
- **测试**: `tests/test_risk_engine.py::test_preprocess_from_artifact_overrides_hardcoded`、`test_legacy_artifact_without_preprocess_still_scores`（旧工件必须继续工作）。
- **依赖**: T5.2（`FEATURE_ORDER` 先定）

### T5.4 评估管线接通 + NEWS2 基线与特征
- **文件**: `scripts/evaluate_model.py`、新建 `scripts/baselines.py`、`app/features.py`（追加特征列）
- **修法**: `evaluate_model.py` 增加 `--input-format labeled_csv`（直接吃 T5.1 产物）；`baselines.py` 实现 `news2_score(row) -> (score, band)`（按 NEWS2 官方权重：RR/SpO2/收缩压/心率/体温，意识项缺测时注明局限）；评估报告并列输出 `model vs news2` 的 AUPRC/AUROC/Brier。
- **NEWS2 特征化**: `news2_score` 同时作为模型输入——`app/features.py` 的 `FEATURE_ORDER` 在 T5.2 七元组特征后追加 `news2_score` 列（缺失时以训练集中位数填补并在 `preprocess` 块记录），把临床金标准分数喂给模型做 stacking 基线；评估报告增加 `model_with_news2` 一列，用增益（AUPRC 差值）决定保留与否，无增益则从 `FEATURE_ORDER` 撤下并记录结论。此变更必须同步写入 T5.3 的工件契约 `feature_names`。
- **测试**: 扩展 `tests/test_evaluation.py`——NEWS2 评分对照手工计算样例；报告含三列指标（`model`、`news2`、`model_with_news2`）；`tests/test_features.py` 断言 `news2_score` 出现在 `FEATURE_ORDER` 且缺失行可计算。
- **验收**: 对 T5.1 demo 数据跑出完整对比 JSON，并附 `model_with_news2` 相对 `model` 的 AUPRC 增益数字。
- **依赖**: T5.1、T5.2

### T5.5 验证工具（时间切分 + bootstrap CI）
- **文件**: 新建 `scripts/validate.py`
- **接口**: `rolling_time_split(rows, n_splits)`、`bootstrap_metric(labels, scores, metric, n=1000) -> (point, lo, hi)`；输出报告格式与 `evaluate_model.py` 一致。
- **测试**: 新建 `tests/test_validation.py`——切分不重叠且按时间有序、95% CI 在合成数据上覆盖率≈95%。
- **依赖**: T5.4

---

## P6 不足版模型（默认版本；500–5,000 阳性为目标区间，demo 数据即可开发）

### T6.1 sklearn 逻辑回归训练管线
- **文件**: 重写 `scripts/train_model.py`（保留 CLI 参数与工件输出形状）
- **修法**: `Pipeline([impute(median, add_indicator), scale, LogisticRegression(C=0.1, penalty="l2", class_weight="balanced")])`；C 用 `rolling_time_split` 网格 `{0.01, 0.1, 1}` 按 AUPRC 选；输出工件含 `preprocess`（T5.3）+ `calibration` 占位；`algorithm` 字段记 `logistic_regression_sklearn`。保留 `--rows` 合成数据演示模式（生成逻辑改为调 T5.1/T5.2 同构管道）。
- **测试**: 新建 `tests/test_training.py`——demo 数据训练产出工件可被 `RiskEngine` 加载且分数在 [0,1]、`preprocess` 块完整。
- **验收**: `python scripts/train_model.py --target-id deterioration_24h` 产出新工件 + `evaluate_model.py` 指标不劣于旧版。
- **依赖**: T5.1–T5.3

### T6.2 校准与阈值反推
- **文件**: `scripts/train_model.py`、新建 `scripts/thresholds.py`、`app/risk.py`
- **修法**: 训练末段在时间校准集上拟合 Platt 标定器，参数写入工件 `calibration: {method: "platt", a, b}`；`_score` 应用标定（缺失该块则跳过，兼容旧工件）。`thresholds.py --target-sensitivity 0.90` 从验证集反推 medium/high/critical 三切点写入工件。
- **测试**: 校准后 Brier ≤ 校准前；阈值脚本在已知分数分布上复现预期切点。
- **依赖**: T6.1、T5.3

### T6.3 promote 门禁接入 manage_model
- **文件**: `scripts/manage_model.py`
- **修法**: `promote` 前强制运行 `evaluate_model.py`（对留出集），门禁 = AUPRC CI 下界 ≥ NEWS2 且 ECE < 0.05，不达标拒绝 promote 并打印差距；报告 JSON 存入 `model-registry/<version>/eval.json`。
- **测试**: 新建 `tests/test_manage_model.py`——低指标工件 promote 被拒、达标通过。
- **依赖**: T6.1、T5.4、T5.5

### T6.4 主动学习标注循环（仅阳性 <500 时启用）
- **文件**: 新建 `scripts/active_label.py`
- **修法**: 输入未标注 landmark CSV + 现役模型 → 按 `|p-0.5|` 升序输出 top-N 待标注清单（含上下文特征值）→ 标注后并回标签集重训。只生成清单，不自动写标签（标注必须人工）。
- **测试**: 清单排序确定性、N 限制生效。
- **依赖**: T6.1

### T6.5 golden dataset 离在线一致性测试（兑现 Review Focus #4 的显式契约）
- **文件**: 新建 `tests/fixtures/golden_dataset.csv`、`tests/test_serving_consistency.py`、`scripts/make_golden.py`
- **背景**: Review Focus #4 要求"`evaluate_model.py` 与 `RiskEngine` 读取同一工件产出相同分数"，但当前只是"隐式契约"，没有任何测试显式钉住它——工件格式（T5.3 preprocess/T6.2 calibration）任何改动都可能让离线指标与在线分数静默分叉。
- **修法**:
  1. `scripts/make_golden.py`：从 T5.1 demo 数据固定抽样 200 行（seed 写死），对指定工件运行 `evaluate_model.evaluate_rows` 与 `RiskEngine.score` 两条路径，将输入与双路径分数写入 `tests/fixtures/golden_dataset.csv`（列：`feature..., label, offline_score, serving_score`）。
  2. `tests/test_serving_consistency.py::test_offline_online_scores_match`：逐行断言 `offline_score == serving_score`（差 < 1e-12），并对两个 target 各跑一遍。
  3. `test_golden_catches_contract_drift`：monkeypatch 工件 `preprocess` 块中的一个 median 值，断言一致性测试变红（防止 fixture 被错误再生掩盖回归）。
  4. CI 增加一步：工件或 `risk.py`/`evaluate_model.py` 变更时，检查 `golden_dataset.csv` 是否需再生（`make_golden.py --check` 模式 diff 输出）。
- **测试**: 即上述测试文件；这组测试就是 Review Focus #4 的落地载体。
- **验收**: `pytest tests/test_serving_consistency.py` 全绿；篡改工件 preprocess 后该测试失败；工件重训后 `make_golden.py --check` 提示需再生 fixture。
- **依赖**: T6.1、T5.3（工件含 preprocess/calibration 后才有分叉风险）

---

## P7 充足版升级（现在实现，合成数据验证；>5,000 阳性仅为真实数据选型参考）

### T7.1 LightGBM challenger + 单调约束
- **文件**: `scripts/train_model.py`（新增 `--algorithm lightgbm` 分支）
- **修法**: 约束向量按 `risk.py:27-38` 临床方向生成（spo2↑→风险↓ 等，写入 `MONOTONE_CONSTRAINTS` 常量）；Optuna 时间 CV 100 trials 目标 = AUPRC − 0.1×ECE；工件新增 `trees` 数组（节点结构 JSON）+ `algorithm: "lightgbm"`。
- **测试**: 约束生效测试（构造 SpO2 递增序列断言分数不升）；工件可加载。
- **依赖**: P6 全部通过且数据量达标

### T7.2 树路径归因
- **文件**: `app/risk.py`（`_score` 按 `algorithm` 分派）
- **修法**: 树模型走路径时累计 `child_value − parent_value` 到分裂特征 → `top_factors` 契约不变；贡献和 ≈ 预测 logit。
- **测试**: 贡献之和与前向计算的 logit 差 < 1e-6。
- **依赖**: T7.1

### T7.3 全门禁与外部验证
- **文件**: `scripts/validate.py`、`scripts/manage_model.py`
- **修法**: 门禁升级——亚组灵敏度差 ≤10%、留一院区验证、决策曲线净获益、isotonic 校准（替换 Platt）。
- **测试**: 合成亚组失衡数据触发门禁失败。
- **依赖**: T7.1、T6.3

### T7.4 漂移监控与再校准调度
- **文件**: `app/observability.py`、新建 `scripts/monitor_drift.py`
- **修法**: 特征 PSI、缺失率、band 占比入 metrics；超阈值（PSI>0.2）写 degraded 告警；月度再校准脚本（只更新 `calibration` 块并走 T6.3 门禁）。
- **测试**: 合成漂移数据触发指标。
- **依赖**: T7.1–T7.3

---

## P8 再入院目标改造（独立子线）

### T8.1 数据契约扩展
- **文件**: `docs/data-contract.md`、`app/domain.py`、`scripts/import_clinical_data.py`
- **动作**: 新增出院时快照字段：`discharge_destination`、`los_days`、`diagnosis_codes[]`、`comorbidity_index`、`prior_utilization_12m`；导入脚本校验。
- **测试**: 缺字段的出院记录被拒并给出字段级错误。
- **依赖**: T5.3（工件契约）

### T8.2 出院触发语义
- **文件**: `app/api.py`、`app/worker.py`
- **动作**: `readmission_30d` 改为出院事件触发单次评分（不再逐事件）；`deterioration_24h` 行为不变。
- **测试**: 出院事件 → 恰好一条再入院评分；非出院事件不触发。
- **依赖**: T8.1

### T8.3 LACE 基线 + 模型训练
- **文件**: `scripts/baselines.py`、`scripts/train_model.py`
- **动作**: 实现 LACE 指数基线；再入院模型（复用 T6.1 管线，特征集 = 新契约字段）必须打赢 LACE 才 promote。
- **测试**: LACE 手工样例对照；门禁拒绝劣于 LACE 的工件。
- **依赖**: T8.1、T6.3

---

## P9 数据自助接入（用户按设计文档自行执行）

> **约定**：P0–P8 已全部实现且不依赖真实数据。本阶段由**用户**按《`docs/data-integration-design.md`》自行完成：准备数据 → 导入 → 再训练 → 过门禁 → 复验。下列任务是给用户的执行清单，不占用开发排期。

### T9.1 数据抽取、脱敏与契约对齐
- **文件**: 新建 `scripts/extract_real_data.py`（占位）、`docs/data-contract.md`、`scripts/import_clinical_data.py`
- **动作**: 与数据方确认抽取口径与治理审批；字段对齐 T5.1 标签契约与 T8.1 出院快照；患者标识脱敏（哈希化）；抽取产物经现有导入校验全量通过。
- **验收**: 抽样数据 `import_clinical_data.py` 导入零失败；治理审批留痕。
- **依赖**: 真实数据源与审批就位

### T9.2 真实数据再训练与门禁复验
- **文件**: `scripts/train_model.py`、`scripts/evaluate_model.py`、`scripts/manage_model.py`、`scripts/make_golden.py`
- **动作**: P5/P6 管线原样接入真实数据训练；过 T6.3 门禁后 promote；再生 golden dataset 并跑 `make_golden.py --check`。
- **验收**: 真实数据评估报告达标；`pytest tests/test_serving_consistency.py` 全绿。
- **依赖**: T9.1、P6 全部

### T9.3 数据量分界与阈值复核
- **文件**: `scripts/thresholds.py`、本文档 P6/P7 分界
- **动作**: 按真实阳性率复核 5,000 分界与 T6.2 目标灵敏度（0.90），必要时修订本文档。
- **依赖**: T9.2

---

## 里程碑

| 里程碑 | 包含 | 完成标志 |
|---|---|---|
| M-A 安全基线 | P0 + P1 | 5 个高危全部有回归测试；默认部署 fail-closed |
| M-B 数据可信 | P2 + P3 | 并发/幂等/对账测试绿；模型加载原子 |
| M-C 产品完整 | P4 | 文档与代码一致；UI 语义正确 |
| M-D ML 可用 | P5 + P6 | 端到端：demo 数据 → 标签 → 训练 → 评估报告（含 NEWS2 对比）→ promote 门禁 |
| M-G 数据自助接入（用户执行） | P9 | 按设计文档完成真实数据接入、再训练过 T6.3 门禁与离在线复验 |
| M-E 数据充足升级 | P7 | 合成数据下 LightGBM 过全部门禁 |
| M-F 再入院立项 | P8 | demo 数据下打赢 LACE |

## 估算与排序说明

- P0+P1：约 1–2 天，**立即执行**。
- P2+P3+P4：约 3–5 天，P2→P3 因共享 `upsert_score`/启动校验而串行，其余可并行。
- P5：约 3–4 天（无真实数据时用 demo 同构管道先跑通）。
- P6：约 2–3 天（demo 数据即可完成开发与验收）。
- P7：约 2–3 天；P8：约 1–2 天（均以合成/demo 数据实现与验证）。
- P9：由用户按《docs/data-integration-design.md》自助执行，不占本计划排期。
- T4.6：约 1 天（需要 Docker Engine 与 `future-infra` profile）；T6.5：约半天；两者随所属阶段交付。
