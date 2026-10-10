# MedIntel 项目面试完全指南

这份指南帮助你把项目讲清楚、讲深入，并能回答常见追问。建议按“项目定位 → 请求链路 → 模型 → 工程可靠性 → 医疗安全 → 局限与演进”学习。

## 一句话定位

MedIntel 是一个面向医院集团的临床风险分层平台。它接收患者住院期间的生命体征和检验事件，聚合截至当前时点可见的特征，分别计算未来 24 小时临床恶化风险和出院后 30 天非计划再入院风险，并把结果送入人工复核、审计和监控闭环。

这个项目不是自动诊断系统。模型负责排序和提醒，医生负责最终判断；权限、数据校验、风险阈值、审计和故障切换由确定性规则完成。

## 30 秒自我介绍版本

> 我做了一个医疗风险分层平台，覆盖临床事件接入、特征快照、双目标模型评分、风险队列、人工复核和审计。开发环境使用 SQLite，生产适配 PostgreSQL；实时链路可通过 Redis Streams 解耦 API 和评分 worker，并支持幂等、重试、pending reclaim 和死信。模型发布支持版本、SHA-256 完整性校验和回滚，认证支持 OIDC/JWT 和租户院区隔离。当前模型基于合成数据，只能用于工程演示和生产前验证，真实临床上线还需要时间外验证、外部院区验证、静默运行和医院审批。

## 3 分钟介绍版本

> 业务问题是医院同时面对两类资源配置问题：住院患者中谁可能在 24 小时内恶化，以及出院患者中谁可能在 30 天内非计划再入院。它们的标签、时间起点、干预动作和代价都不同，因此我把它们建成两个独立 target，各自拥有模型文件、版本和阈值。
>
> 数据以 ClinicalEvent 进入系统。API 先完成数值范围、UTC 时间、患者与就诊关系、租户和院区范围校验，再依靠 event_id 保证幂等。评分前，系统按 occurred_at 聚合该次就诊截至当前时点最近可用的特征，避免使用未来事件造成数据泄漏。开发模式可以同步评分；生产模式可以把事件发布到 Redis Streams，由 consumer group 中的 worker 异步评分。成功后 ACK，失败消息保留在 pending 中，worker 可以 reclaim，超过最大次数进入死信流。
>
> 模型层使用可解释逻辑回归形成完整的训练、加载、评分和评估切片。评估包含 AUROC、AUPRC、Brier、灵敏度、特异度、PPV、时间切分和亚组结果。生产模型需要带 target、特征签名、阈值和版本，发布工具生成 SHA-256 manifest；摘要不一致时拒绝启动。默认采用 fail-closed，显式启用降级时才输出带 model_unavailable 标记的不可行动结果。
>
> 平台层还有 PostgreSQL JSONB 和连接池、OIDC/JWT 与 JWKS 验签、RBAC、租户隔离、Prometheus 指标、健康检查和人工处置状态保留。这个项目已经完成工程闭环，但临床有效性不能由合成数据证明，真实上线仍需医院数据治理、独立验证、静默试点和多方签字。

## 总体架构

```mermaid
flowchart LR
    HIS[HIS/EMR/LIS/设备] --> API[FastAPI 事件入口]
    API --> VAL[契约与权限校验]
    VAL --> DB[(SQLite / PostgreSQL)]
    VAL -->|sync| SCORE[双目标风险引擎]
    VAL -->|redis| STREAM[(Redis Streams)]
    STREAM --> WORKER[评分 Worker]
    WORKER --> SCORE
    SCORE --> DB
    DB --> UI[风险工作台]
    UI --> REVIEW[人工复核与处置]
    REVIEW --> DB
    API --> AUDIT[审计]
    API --> METRICS[健康与指标]
```

面试时从左到右讲：数据从哪里来，怎样保证合法和隔离，如何评分，怎样可靠传输，结果如何被人使用，出了问题怎么发现和恢复。

## 目录与职责

| 文件 | 作用 | 面试关键词 |
| --- | --- | --- |
| `app/main.py` | 创建应用、生命周期、开发数据初始化、HTTP 指标 | dependency injection、lifespan、环境隔离 |
| `app/domain.py` | Pydantic 领域模型和数据约束 | schema、UTC、临床数值范围 |
| `app/api.py` | API、鉴权依赖、事件接入、查询、复核、审计 | idempotency、scope、sync/async |
| `app/risk.py` | 模型加载、签名验证、特征变换、评分和解释 | logistic regression、fail-closed |
| `app/store.py` | SQLite 开发存储 | WAL、事务、复合键、迁移 |
| `app/postgres_store.py` | PostgreSQL 生产适配器 | pool、JSONB、durability |
| `app/storage.py` | 根据配置创建存储 | adapter/factory |
| `app/messaging.py` | Redis Stream 发布、消费、重试和死信 | consumer group、ACK、pending、DLQ |
| `app/worker.py` | 异步评分和批量重算 | eventual consistency、retry |
| `app/security.py` | 演示令牌、OIDC/JWT、RBAC、范围映射 | JWKS、issuer、audience、least privilege |
| `app/observability.py` | Prometheus 文本指标 | counters、latency、health |
| `scripts/train_model.py` | 生成演示模型 | artifact contract、reproducibility |
| `scripts/evaluate_model.py` | 时间切分和模型指标 | discrimination、calibration、subgroup |
| `scripts/manage_model.py` | 模型提升、摘要清单和回滚 | registry、integrity、rollback |

## 两个预测目标

### 未来 24 小时临床恶化

索引时点是在住院期间每次有新临床事件时。标签表示之后 24 小时内是否发生预先定义的恶化事件，例如 ICU 转入、通气、升压药、心肺复苏或院内死亡。真实项目必须由临床团队冻结事件集合和排除规则。

### 出院后 30 天再入院

索引时点应是出院时。标签表示之后 30 个自然日内是否发生非计划再入院。这里的“30 天”是窗口内二分类：回答会不会发生，不能回答具体第几天发生。

如果需要 7、14、21、30 天曲线，可以有两种设计：

1. 每个窗口训练独立二分类模型，易部署和解释，但多个概率可能不满足单调关系。
2. 使用生存分析或离散时间 hazard 模型，直接预测事件时间分布，更适合回答“什么时候发生”，但训练、删失处理和临床解释更复杂。

当前项目固定了两个 `TargetId`。继续扩展时间窗时，需要同步扩展领域类型、模型配置、训练评估、数据库查询和前端选择器。

## 一次事件是怎样处理的

以一个乳酸结果为例：

```json
{
  "event_id": "evt-1001",
  "event_type": "lab_result",
  "occurred_at": "2026-09-09T08:30:00Z",
  "tenant_id": "group-demo",
  "facility_id": "hospital-01",
  "patient_id": "pat-001",
  "encounter_id": "enc-001",
  "source_system": "LIS",
  "source_record_id": "lis-88721",
  "unit": {"lactate": "mmol/L"},
  "quality_status": "valid",
  "payload": {"lactate": 3.2}
}
```

处理顺序：

1. Pydantic 验证字段类型，乳酸必须是有限数并落在配置的临床合理范围内。
2. 把无时区时间解释为 UTC，有时区时间转换为 UTC；超过允许时钟偏差的未来事件被拒绝。
3. JWT 映射出用户角色、tenant 和 facility，请求里的范围必须和身份范围一致。
4. 验证 encounter 确实属于 patient、tenant 和 facility。
5. 按 `event_id` 插入数据库。相同内容的重复请求安全返回；同 ID 不同内容返回 409。
6. 同步模式直接生成特征快照并评分；Redis 模式发布事件，由 worker 完成评分。
7. 两个 target 分别写入分数，键是 `(tenant_id, encounter_id, target_id)`。
8. 写审计和指标，工作台随后读取最新分数。

### 为什么不能直接用本次 payload 评分

临床事件通常是部分更新。一次化验可能只有乳酸，没有年龄、血氧或既往住院次数。`feature_snapshot` 会按事件时间遍历该次就诊已有事件，把截至评分时点每个字段最近的非空值合并成特征快照。

`as_of` 很关键。离线回放 8:30 的事件时，不能把 10:00 才出现的化验值带入模型，否则会形成未来数据泄漏，离线指标会虚高。

## 模型原理

当前模型是逻辑回归：

```text
z = bias + Σ(weight_i × transformed_feature_i)
p = 1 / (1 + exp(-z))
```

系统先对部分特征做临床方向变换：

- 年龄使用 `max(age - 50, 0)`；
- 收缩压以 120 为中心；
- 血氧以 96 为中心；
- 体温使用距离 37°C 的绝对偏差；
- 其他特征按原值进入。

得到概率后，再根据每个 target 自己的 `medium/high/critical` 阈值转为风险等级。`top_factors` 按绝对贡献值排序，帮助临床人员理解哪些输入推动了分数。

### 为什么选逻辑回归

面试回答：首个生产前切片更重视可解释、低延迟、可校准和稳定部署。逻辑回归可以提供清晰的特征贡献，依赖少，易于做基线和监控。它不代表永远是最佳算法；真实数据到位后应比较逻辑回归、梯度提升树和时间序列模型，并使用时间外验证、校准、告警负荷和净获益选择最终模型。

### 为什么两个目标不能共用一个输出

恶化和再入院有不同的索引时点、标签窗口、患病率、误报代价、可用特征和干预流程。即使采用共享底座，也必须有独立输出头、阈值、版本、验证报告和审批记录。当前项目选择两个独立 artifact，使治理关系最清晰。

## 指标必须会讲

| 指标 | 含义 | 面试重点 |
| --- | --- | --- |
| AUROC | 随机抽一名阳性和阴性，模型把阳性排得更高的概率 | 衡量排序，不反映概率是否可信；类别极不平衡时可能显得乐观 |
| AUPRC | 精确率—召回率曲线下面积 | 更关注少数阳性，医疗预警通常比 AUROC 更有信息 |
| Brier | 概率和真实标签之间的均方误差 | 同时反映概率误差，越小越好 |
| Sensitivity | `TP/(TP+FN)` | 找出多少真正高风险患者；漏报代价高时重要 |
| Specificity | `TN/(TN+FP)` | 排除多少低风险患者 |
| PPV | `TP/(TP+FP)` | 发出的告警中有多少是真的，直接影响告警疲劳 |

阈值不能只按最高 AUROC 选择，因为 AUROC 与阈值无关。医院需要结合床位、人力、漏报成本、每日告警数和处置能力选择阈值。高风险患者只有在存在可执行干预时，预警才有业务价值。

### 数据切分

随机切分容易把同一患者或同一时期的模式泄漏到测试集。医疗场景至少需要：

- 患者级隔离；
- 时间外测试；
- 外部院区验证；
- 年龄、性别、科室、院区、设备等亚组报告；
- 上线后的校准、漂移和告警负荷监控。

## 数据库设计

SQLite 用于本地开发，开启 WAL、busy timeout、外键和进程内锁。PostgreSQL 适配器用于生产，使用连接池、TIMESTAMPTZ 和 JSONB。

关键键设计：

- patient/encounter：`(tenant_id, encounter_id)`；
- score：`(tenant_id, encounter_id, target_id)`；
- event：当前使用全局唯一 `event_id`；
- audit：独立 `audit_id`。

为什么 score 必须包含 `target_id`：同一就诊同时有两个风险结果和两套人工复核状态。如果只以 encounter 为键，第二个模型会覆盖第一个模型。

为什么 JSONB：临床字段和风险解释可能快速演进，JSONB 保留灵活性，结构化列负责范围、关联和索引。高频查询字段仍应拆成普通列并建立索引，不能把所有查询都压在 JSONB 上。

## 幂等与一致性

幂等意味着调用方可以安全重试。项目用 `event_id` 去重：

- 第一次事件插入并处理；
- 完全相同的事件再次到达，返回 duplicate；
- 相同 ID 但内容不同，返回 409，避免静默覆盖。

Redis worker 采用 at-least-once 语义，所以同一事件可能被处理多次。数据库事件幂等和分数 upsert 使重复处理不会生成无限重复记录。

要诚实说明当前边界：数据库提交和 Redis 发布不是一个跨系统原子事务。发布失败时事件已经落库，客户端重试会重新发布；更严格的生产实现应增加 transactional outbox，由独立 dispatcher 可靠投递并监控 outbox backlog。

## Redis Streams 可靠性

核心概念：

- stream 保存事件；
- consumer group 在多个 worker 之间分配消息；
- 处理成功后 ACK；
- worker 崩溃时消息留在 pending entries list；
- 空闲超过阈值后由其他 worker `XAUTOCLAIM`；
- 多次失败的 poison message 进入 dead-letter stream。

选择 Redis Streams 的理由是项目体量小、部署简单、支持 consumer group 和 pending 恢复。如果吞吐、长期保留、多下游消费和跨机房能力显著增加，可以换 Kafka；API 契约、事件 envelope 和 worker 边界保持不变。

## 认证、授权和租户隔离

开发环境使用三个显式 Bearer token。生产环境启用 OIDC：

1. 通过 JWKS 获取公钥并验证 JWT 签名；
2. 检查 issuer、audience、exp 和 sub；
3. 从配置的 claim 提取角色、tenant 和 facility；
4. 把角色映射为最小权限集合；
5. 每次查询和写入都带 tenant/facility 过滤。

RBAC 回答“能做什么”，tenant/facility scope 回答“能操作哪些数据”。两者缺一不可。管理员可以跨院区，但仍受 tenant 限制；clinician 必须绑定 facility。

需要承认：真实医院还要处理密钥轮换、JWKS 缓存策略、用户离职撤权、break-glass、细粒度科室权限、审计留存和渗透测试。

## 人工复核闭环

风险不是最终结论。每个 target 的 score 保存 `new / acknowledged / in_progress / resolved` 和备注。worker 重算时会更新概率、等级和模型版本，但保留已存在的人工状态和备注，防止新数据把临床处置记录覆盖掉。

这体现 human-in-the-loop：

- 模型负责发现和排序；
- 确定性阈值决定进入哪个工作队列；
- 临床人员判断是否需要处置；
- 审计记录谁在何时做了什么。

## 模型发布、签名和回滚

模型 artifact 至少包含：

- `target_id`；
- `model_version`；
- `feature_names`；
- 权重和 bias；
- 阈值；
- 算法与标签定义。

`manage_model.py promote` 验证 target、版本和特征签名，把模型复制到 target/version 目录，计算 SHA-256，并更新 manifest。服务启用签名要求后会重新计算摘要；文件被篡改或复制错误时拒绝启动。rollback 从已有版本恢复 serving artifact 并更新 active manifest。

SHA-256 只能证明文件与清单一致。生产还应对清单做数字签名，把审批人、训练数据版本、代码 commit、评估报告和发布时间绑定到模型版本。

## 故障与降级

医疗场景默认 fail-closed：模型缺失、格式错误或签名不一致时，生产服务拒绝启动，避免悄悄使用一套未经批准的默认权重。

如果医院明确批准降级模式，系统会返回零分，同时加入 `model_unavailable` 和人工处理提示，健康接口返回 `degraded`。这个结果不能被当成“低风险”。实际工作台和告警路由应对该标记做醒目阻断。

## 可观测性

`/api/health` 显示环境、模型版本、降级模型和 Redis 可用性；`/api/metrics` 输出 Prometheus 格式的 HTTP 状态、延迟、事件接收、拒绝、重复、排队、评分、复核和错误计数。

生产还应该增加：

- 数据延迟和特征缺失率；
- 每个 target 的分数/风险等级分布；
- Redis lag、pending 和 DLQ 数量；
- 模型校准、漂移和分群性能；
- 每日告警数、确认时间和处置完成率；
- API/worker SLO、数据库连接池和容量告警。

## 为什么不是每个环节都用机器学习

这是面试加分点。机器学习适合解决依赖历史统计规律的预测和排序问题。以下环节使用规则更可靠：

- schema 和数值范围校验；
- 权限和租户隔离；
- 幂等、事务和重试；
- 风险等级阈值；
- 审计与合规保留；
- 最终临床处置。

“企业级机器学习”并不是把 ML 塞进所有模块，而是在预测收益高的环节使用 ML，并用确定性系统约束它。

## 演示顺序

### 1. 启动

```powershell
& .\.venv\Scripts\uvicorn.exe app.main:app --reload
```

打开 `http://localhost:8000` 和 `http://localhost:8000/docs`。

### 2. 查询风险队列

```powershell
Invoke-RestMethod http://localhost:8000/api/patients `
  -Headers @{Authorization='Bearer demo-clinician-token'}
```

### 3. 发送事件

```powershell
$body = @{
  event_id='interview-event-1'
  event_type='lab_result'
  patient_id='pat-001'
  encounter_id='enc-001'
  tenant_id='group-demo'
  facility_id='hospital-01'
  source_system='LIS'
  source_record_id='demo-1'
  quality_status='valid'
  unit=@{lactate='mmol/L'}
  payload=@{lactate=4.2}
} | ConvertTo-Json

Invoke-RestMethod http://localhost:8000/api/events -Method Post `
  -ContentType 'application/json' -Body $body `
  -Headers @{Authorization='Bearer demo-clinician-token'}
```

重复发送展示幂等，再把乳酸改成 4.5 且保留相同 event_id，展示 409 冲突。

### 4. 切换目标和人工复核

先投递一条出院事件——再入院分数只由出院事件触发（demo 种子不再预置，T8.2 语义）：

```json
{
  "event_id": "evt-discharge-demo",
  "event_type": "discharge",
  "occurred_at": "2026-09-09T10:00:00Z",
  "patient_id": "pat-001",
  "encounter_id": "enc-001",
  "payload": {
    "discharge_destination": "home",
    "los_days": 4.0,
    "diagnosis_codes": ["I50.9"],
    "comorbidity_index": 1.0,
    "prior_utilization_12m": 0,
    "ed_visits_6m": 1
  }
}
```

再请求 `/api/patients?target_id=readmission_30d`（此时应能看到 enc-001 的再入院分数），然后向 `/api/patients/enc-001/review` 提交：

```json
{
  "target_id": "readmission_30d",
  "status": "acknowledged",
  "note": "已进入出院随访计划"
}
```

### 5. 展示治理

用 quality token 查看 `/api/model-card` 和 `/api/audit`，再打开 `/api/health` 与 `/api/metrics`。

## 高频追问与参考回答

### 1. 为什么用事件驱动架构？

生命体征和化验会持续变化，批量每天计算无法及时发现恶化。事件驱动可以在新数据到达时更新风险，并通过消息队列削峰、重试和扩展 worker。再入院目标也可以在出院事件到达时触发。

### 2. 为什么 API 返回 202？

202 表示事件已接受但异步评分未必完成，符合 Redis 模式下的最终一致性。同步开发模式也保持同一接入语义，调用方不用因部署方式变化而更换接口。

### 3. 如何处理乱序事件？

事件保留 `occurred_at`，快照按事件时间排序并支持 `as_of`。离线回放不会读取未来值。生产还应定义迟到窗口；严重迟到时触发受控重算，并监控迟到率。

### 4. 如何处理重复事件？

用 event_id 保证幂等。相同事件安全忽略，同 ID 异内容返回冲突。异步 worker 采用 upsert，因此重复消费不会重复生成风险行。

### 5. Redis 挂了会怎样？

API 已持久化事件，发布失败返回 503 并写审计；调用方可用相同 event_id 重试。worker 读取失败会退避重试，健康接口显示 event bus unavailable。更严格方案是 transactional outbox。

### 6. worker 处理一半崩溃怎么办？

消息未 ACK 会保留在 pending。超过 claim idle 后，存活 worker 可以 reclaim。评分写入是 upsert，所以重新处理安全。

### 7. 为什么需要死信流？

格式正确但始终无法处理的数据会阻塞和消耗资源。超过投递次数后转移到 DLQ，保留原始事件和错误，供修复后重放。

### 8. 如何防止跨租户数据泄漏？

身份声明提供 tenant/facility，API 校验请求范围，数据库每个患者、事件、分数和审计查询都带范围条件，核心表使用租户复合键。测试覆盖同 encounter_id 在不同 tenant 不互相覆盖。

### 9. 为什么 PostgreSQL 还存 JSONB？

JSONB 方便事件和解释结构演进，关系列承担主键、租户范围、时间和索引。真实高频分析字段要正规化或进入特征平台，避免全表扫描 JSONB。

### 10. 怎样避免训练—服务偏差？

固定特征顺序和变换逻辑，把 `feature_names` 写入 artifact 并在加载时验证；更完整方案是把预处理器和模型共同打包，使用同一特征定义代码，增加 golden dataset 和离在线一致性测试。

### 11. 为什么不能只看 AUROC？

AUROC 不表示概率校准，也不直接体现阳性稀少时的告警质量。还要看 AUPRC、Brier、校准曲线、灵敏度、PPV、每天告警数、处置能力和决策净获益。

### 12. 模型漂移怎么监控？

先监控输入分布、缺失率和来源变化，再监控分数分布；标签成熟后计算时间窗口性能和校准。按院区与亚组分层，对偏移设置阈值并进入调查、再校准或回滚流程。

### 13. 如何选择阈值？

在验证集上形成不同阈值的灵敏度、PPV 和每日告警量表，由临床与运营团队结合漏报成本和实际处置容量选择。阈值必须按 target 单独审批。

### 14. 为什么模型分数不能直接触发治疗？

输入可能缺失或错误，模型存在分布外和误判风险，而且治疗决策需要完整病史和临床判断。该系统定位为风险排序和提醒，处置由授权临床人员完成并审计。

### 15. 如何回滚模型？

模型按 target/version 保存。发布工具验证 artifact、生成摘要并更新 active manifest；异常时选择已验证旧版本，复制回 serving path 并重新启动或滚动替换。生产还应配灰度、双跑和自动 SLO 门控。

### 16. 为什么 SHA-256 还不等于安全签名？

如果攻击者能同时修改模型和 manifest，摘要仍会匹配。真实生产应由受信任 CI 或密钥对 manifest 做数字签名，并限制 artifact store 写权限。

### 17. 系统是一致还是最终一致？

同步模式在请求内完成评分；Redis 模式在事件持久化后异步评分，是最终一致。事件、分数和人工状态分别通过幂等与 upsert 保持可恢复。

### 18. 这个项目最大的技术取舍是什么？

为了让完整链路可运行，模型和基础设施都选择了轻量实现：逻辑回归、SQLite 和文件 registry。同时通过 adapter 保留 PostgreSQL、Redis、OIDC 和受控模型发布边界。代价是还缺少真正的 outbox、集中式 Registry、线上漂移作业和医院级 HA 验证。

### 19. 真实上线第一件事做什么？

先与临床团队冻结 cohort、索引时点、标签和干预 SOP，再做数据映射与质量画像。没有可靠标签和可执行干预，继续优化算法没有意义。

### 20. 你如何证明项目质量？

自动化测试覆盖 API、鉴权、租户隔离、事件幂等、时间快照、双目标、人工状态保留、模型签名、降级和模型回滚；CI 执行 pytest、Ruff 和 compileall。生产依赖与真实 PostgreSQL/Redis/OIDC 仍需在集成环境做容器和故障注入测试。

## 面试中要主动承认的局限

主动说清局限会比夸大项目更加分：

1. 模型由合成数据训练，没有临床有效性结论。
2. 当前是固定特征逻辑回归基线，还没有真实数据上的候选模型比较和校准图。
3. PostgreSQL、Redis 和真实 OIDC 适配器已有代码，但本地单元测试环境没有完成真实医院配置下的端到端验证。
4. 数据库与 Redis 之间还不是 transactional outbox，极端故障下依赖重试恢复。
5. 内置 metrics 是基础版，生产需要集中抓取、标签维度、SLO、漂移与业务闭环看板。
6. 文件 registry 可演示版本治理，医院生产应接入有权限、审批和数字签名的模型仓库。
7. 临床上线需要伦理/质量/信息安全/医疗器械相关评估，代码不能代替这些证据。

## 不要这样说

- 不要说“模型可以诊断患者”。说“模型提供风险分层，辅助人工复核”。
- 不要说“30 天模型能预测第几天再入院”。它预测 30 天窗口内是否发生。
- 不要说“通过 28 个测试就能上线”。测试证明代码行为，不能证明临床有效性。
- 不要说“SHA-256 防止所有攻击”。它验证完整性，信任来源还需要签名和权限。
- 不要说“Redis 保证 exactly-once”。这里是 at-least-once，通过幂等实现业务效果稳定。
- 不要说“所有环节都用了 ML”。校验、权限、审计和最终决策更适合规则和人工。

## 项目亮点的 STAR 表达

**情境**：医院数据来自多个系统，更新频繁，既要尽早发现高风险患者，又必须避免跨院区泄漏和模型错误直接影响诊疗。

**任务**：实现一个可运行的生产前纵向切片，覆盖双目标风险预测和人工闭环，同时保留生产数据库、消息、身份和模型治理边界。

**行动**：设计统一事件契约和时间快照；用 target 隔离模型与复核状态；实现租户复合键、OIDC scope、Redis consumer group、幂等与死信；加入模型签名、回滚、健康、指标和临床试点协议。

**结果**：形成从事件到处置审计的完整闭环，通过自动化测试和静态检查，可用于接口联调、演示和真实数据接入前验证。临床有效性保持为待医院真实数据验证，不夸大结果。

## 七天学习计划

### 第 1 天：业务与架构

背熟 30 秒和 3 分钟介绍，能不看图画出数据流，解释两个 target 为什么分开。

### 第 2 天：事件与数据库

亲自发送正常、重复、冲突、越权、未来时间和异常数值事件。讲清幂等、复合键和特征快照。

### 第 3 天：模型与指标

手算一个简单逻辑回归概率，讲清 AUROC、AUPRC、Brier、灵敏度、PPV 和阈值取舍。

### 第 4 天：Redis 与一致性

画出 publish、read、pending、reclaim、ACK、DLQ。讲清 at-least-once、upsert 和 outbox 缺口。

### 第 5 天：安全与治理

讲清 JWT 签名、issuer、audience、JWKS、RBAC、tenant scope、模型摘要和回滚。

### 第 6 天：医疗落地

讲 cohort、索引时点、标签、时间切分、亚组、公平性、静默运行、告警疲劳和 human-in-the-loop。

### 第 7 天：模拟面试

录制一次 3 分钟介绍和一次 15 分钟深挖。回答上面的 20 个问题，每个答案控制在 30 到 90 秒。

## 掌握标准

当你能完成下面六件事，就足以应对多数后端、ML 工程和 MLOps 面试：

1. 三分钟内完整讲清问题、架构、结果和边界。
2. 画出同步与 Redis 两条事件链路。
3. 解释时间泄漏、幂等、at-least-once 和租户隔离。
4. 解释逻辑回归、六个评估指标以及阈值如何由业务容量决定。
5. 解释模型发布、完整性校验、降级和回滚。
6. 主动指出真实临床验证、outbox 和基础设施集成测试等局限，并给出合理演进方案。

## 简历写法

根据岗位选择三到四条，不要把所有技术堆进一句话：

- 设计并实现面向医院集团的双目标风险分层平台，覆盖临床事件接入、时间点特征快照、风险评分、人工复核与审计闭环。
- 使用 FastAPI、PostgreSQL 与 Redis Streams 构建事件驱动评分链路，通过 event ID 幂等、consumer group、pending reclaim、DLQ 和 upsert 支持 at-least-once 消费。
- 实现 OIDC/JWT、JWKS 验签、RBAC 以及 tenant/facility 数据范围控制，以复合主键和查询过滤降低跨租户泄漏风险。
- 建立模型 artifact 契约、target 独立版本、SHA-256 完整性校验、发布回滚和显式降级机制，并提供时间外与亚组评估工具。
- 为 API、评分、队列和人工复核加入健康与 Prometheus 指标，以自动化测试覆盖幂等、隔离、时间泄漏、双目标和模型治理行为。

不要在简历中填写不存在的线上规模、临床指标或医院名称。没有真实数据时，不写“准确率达到 95%”或“已服务十万患者”。

## 白板讲解模板

面试官让你画架构时，按下面顺序画，避免一上来堆组件：

1. 先画 HIS/EMR/LIS → API → 风险结果 → 临床工作台，说明业务闭环。
2. 在 API 后加契约校验、OIDC 和 tenant/facility scope，说明安全边界。
3. 在评分前加 PostgreSQL 和特征快照，强调 event time 与 `as_of`。
4. 在 API 与 worker 之间加 Redis Streams，说明 202 和最终一致性。
5. 在 worker 周围标出 ACK、pending、reclaim、DLQ 和幂等 upsert。
6. 在模型旁标出两个 target、版本、阈值、manifest 和 rollback。
7. 最后补 health、metrics、audit 和 human review。

讲完主动指出两个缺口：数据库和 Redis 之间应进一步增加 transactional outbox；真实模型需要医院数据上的时间外、外部院区和前瞻性静默验证。

## 临场答题框架

遇到架构题，用“需求—约束—方案—故障—取舍”回答：

1. **需求**：要预测谁有风险，结果给谁使用，多久需要更新。
2. **约束**：医疗数据敏感、标签延迟、类别不平衡、不能让模型直接决策。
3. **方案**：事件契约、特征时点、模型、存储、消息和人工闭环。
4. **故障**：重复、乱序、队列中断、worker 崩溃、模型损坏、身份越权。
5. **取舍**：为什么当前选择逻辑回归、Redis、JSONB，以及规模扩大后的替代方案。

遇到算法题，用“标签—切分—基线—指标—阈值—监控”回答。先定义正确的预测问题，再讨论算法；先防止患者和时间泄漏，再比较模型；最后把阈值落到每日告警量和处置容量。

## 90 分钟速成顺序

如果面试就在当天：

- 0–15 分钟：背熟 30 秒和 3 分钟介绍。
- 15–30 分钟：画两遍总体架构和一次事件链路。
- 30–45 分钟：掌握逻辑回归、AUROC、AUPRC、Brier、Sensitivity 和 PPV。
- 45–60 分钟：掌握幂等、at-least-once、pending reclaim、DLQ 和 outbox。
- 60–75 分钟：掌握 OIDC/JWT、租户隔离、模型签名、降级和回滚。
- 75–90 分钟：回答高频问题 2、4、5、8、10、11、13、18、19、20。
