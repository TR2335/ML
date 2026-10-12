# MedIntel 机器学习项目演化流程

> 本文档记录项目从初始提交到当前状态的完整演化路径，以及后续演化方向。
> 数据来源：git 历史（58 个提交）+ `docs/superpowers/plans/` 各阶段实施计划。更新于 2026-10-10。

## 1. 项目愿景与定位

**目标**：面向医院集团的企业级 ML 平台，首个纵向切片覆盖两个预测目标——

| 目标 | 触发 | 含义 |
| --- | --- | --- |
| `deterioration_24h` | 逐临床事件 | 未来 24 小时临床恶化风险 |
| `readmission_30d` | 仅出院事件（单次） | 出院后 30 天非计划再入院风险 |

**定位**：研究与生产前验证骨架——ML 负责有历史信号的排序，规则/人工复核/运营约束保留在决策链上。不是医疗器械软件，不单独改变诊疗方案。

## 2. 演化总览

```mermaid
flowchart LR
  P0["P0 工程基线<br/>09-18"] --> P1["P1 高危修复<br/>09-27"]
  P1 --> P2["P2 数据完整性与并发<br/>09-28"]
  P2 --> P3["P3 模型与配置<br/>09-28"]
  P3 --> P4["P4 API/前端/文档<br/>09-28"]
  P4 --> P5["P5 ML 数据管线<br/>09-28"]
  P5 --> P6["P6 不足版模型<br/>10-10"]
  P6 --> P8["P8 再入院目标改造<br/>10-10"]
  P8 --> P7["P7 充足版升级<br/>10-10"]
  P7 --> NOW["当前：已推送同步<br/>156 tests 通过"]
  NOW --> OPT["优化轮（进行中）<br/>患者级切分+报告卡+KS漂移+2.5w规模演练"]
  OPT --> P9["P9 真实数据自助接入"]
  P9 --> PROD["生产化演进<br/>FHIR 适配 / OIDC / Prometheus / 前瞻性评估"]
```

**方法论**：每个阶段遵循 `需求拆解（WBS）→ writing-plans 写终版计划 → TDD 逐任务独立提交 → 全新上下文终审 → 唯一修复轮`。主 WBS 见 `docs/superpowers/plans/2026-09-27-medintel-task-breakdown.md`。

## 3. 阶段详情

### P0 · 工程基线（2026-09-18，提交 `7ddab7c`）

初始骨架一次成型：

- FastAPI 应用（事件接入、风险评分、队列查询、人工复核、审计）+ SQLite WAL 存储 + 纯 Python 逻辑回归风险引擎
- 异步 worker（事件触发重评分）、演示 RBAC、Prometheus 指标、轻量风险工作台前端
- Docker Compose、README、模型卡/架构/契约文档

### P1 · 高危修复（2026-09-27，10 个提交 `4938800..c6521ad`）

安全与数据一致性高危清零：

- 生产环境演示认证 fail-closed；前端输出全量转义 + CSP 头；事件身份租户级 UUID
- 分数写入跨进程原子事务；`as_of` 陈旧性守卫；Redis worker 从持久事件对账缺失分数
- 导入共享未来时间校验且单事务；复核与重评分状态语义分离

### P2 · 数据完整性与并发（2026-09-28，提交 `572b924`、`285dbae` 等）

- 租户迁移幂等且原子；同步 worker 失败退避 + 重启/健康检查
- 集成测试套件初建（`1ca4dcb` 跨后端一致性，标记排除默认运行）

### P3 · 模型与配置（2026-09-28，提交 `979a3ee..69d1a12`）

- 模型签名按环境判定（dev 关闭、非 dev 强制 fail-closed）
- 工件原子加载 + 缺失原因上浮；生产配置硬化（禁游离 `.env`、键名拼写检查）
- 审计 actor/resource 院区范围拆分

### P4 · API / 前端 / 文档（2026-09-28，提交 `6f98546..afc3899`）

- 患者列表分页、端点披露收敛、幂等比较收紧、指标标签净化
- 跨目标分数回退移除；文档与代码对齐（目录树、故障矩阵、配置说明）
- 低危清理批次 + 集成套件隔离修复

### P5 · ML 数据管线（2026-09-28，提交 `14968a2..ea092b7`）

从平台走向 ML 的转折点：

- **结局表型**：landmark 标签派生（demo 结局管道）+ 契约文档
- **七元组特征**：`build_features.py` 共享特征顺序，训练/serving 同源
- **工件 v2**：preprocess + `feature_order` 契约
- **NEWS2 基线**三方对比评估；**滚动时间切分 + bootstrap 置信区间**（重复时间戳不跨折）

### P6 · 不足版模型（2026-10-10，提交 `0455502..f610305`）

模型从"能算"到"能管"：

- sklearn 管线训练（中位数填补 → 标准化 → L2，C 按时间 CV AUPRC 选取）
- Platt 校准 + 灵敏度阶梯阈值；**promote 门禁 v1**（bootstrap AUPRC 下界 ≥ NEWS2 + ECE<0.05）
- 不确定性主动标注队列（只排队不写标签）；golden 数据集钉住离线/serving 分数一致性
- 终审修复（hermetic golden、promote 白名单、目标域加载）

### P8 · 再入院目标改造（2026-10-10，提交 `4606a8e..484bf6a` + `831602b`）

`readmission_30d` 从"有"到"对"：

- 出院事件强制 6 字段快照；**仅出院事件触发**单次再入院评分（`as_of` 钉首个出院）
- LACE 基线（Austin 2010 分箱）；再入院模型切换到出院快照特征空间
- promote 门禁对 LACE；文档与特征空间描述对齐；终审 0C+3I 修复

### P7 · 充足版升级（2026-10-10，提交 `7284216..2f8ae51`）

生产级 ML 控制齐备（执行序在 P8 后交错）：

- **LightGBM opt-in 挑战者**（单调约束 + 漂移参考段写入）
- **纯 Python 树打分**：路径行走 + 归因，serving 零依赖；树工件严格校验（环/越界/负索引响亮拒绝）
- **校准升级**：isotonic 替换 Platt（低分辨率自动回退 R-P7-5a，保留 legacy 支持）
- **门禁 v3**：亚组灵敏度差 ≤10% + 逐院区 AUPRC ≥ 基线 + 决策曲线平均增益 ≥0
- **漂移监控**：训练写 `drift_reference`；`monitor_drift.py` PSI>0.2 告警 + `recalibrate.py` 门禁内再校准
- 终审 0C+4I 修复

### P-Scale · 真实规模优化（2026-10-12，设计 spec `docs/superpowers/specs/2026-10-12-medintel-real-scale-optimization-design.md`）

面向 2–3 万出院/入院体量的方法学与工程验证（P9 真实数据前置必修），方案 A"并行增量"，演示路径零改动：

- **G1 规模队列**：`app/cohort.py`（纯标准库确定性合成：22% 多次住院、4% 恶化事件率 + 事件前漂移、12% 再入院 latent、MAR 缺失 5–12%）+ `scripts/generate_scale_data.py` 流式写 16 格 landmark 数据集
- **G2 患者级防泄漏**：`patient_rolling_time_split`（整患者不跨折 + 患者时点不拆块 + 返回前自证交集为空）；`validate`/`evaluate`/`promote` 三处 `--split`（默认 patient，缺分组键响亮报错）
- **G3 报告卡**：`scripts/report_card.py` 10 章节（md+json），指标只复用既有函数（discrimination 与直接 `binary_metrics` 等值测试钉住）
- **G4 漂移双检**：`compute_ks`（纯排序双指针，效应量不看 p 值）+ 训练写确定性参考样本（≤2000/特征）；readmission 派生特征输入缺列时即时编码
- **G5 规模演练**：`scripts/scale_rehearsal.py` 编排 7 组 13 步（生成→训练→患者级验证→评估→门禁→报告卡→漂移自检），分组时间预算 + 机读通过标准
- **性能前置**：`binary_metrics` AUPRC O(n²)→O(n log n)（整数前缀和位级等价）；bootstrap 规模预算（>50000 行 → 500 次 × ≤20000 行子采样，报告注明 `replicate_subsample`）

### 当前状态（2026-10-12）

- 本轮 8 任务（T1–T8）逐任务 TDD 独立提交；25000 规模端到端演练真跑见 `data/rehearsal_report.json`（`data/` 不入库，关键数字见 STATUS.md）
- 验证基线：`pytest` 194 passed + 1 skipped + 2 deselected（含 1500 规模冒烟）；`ruff` 全零；`app/` serving 纯净（`app/cohort.py` 仅标准库）
- 24 项 P6–P8 终审 deferred minors 清单仍在 `docs/superpowers/STATUS.md`（本轮新增裁定与 deferred 见该文件）

## 4. 横切演化主线

| 主线 | 演化路径 |
| --- | --- |
| 模型 | 手写逻辑回归（P0）→ sklearn 管线 + CV 选 C（P6）→ LightGBM 挑战者 + 纯 Python 树打分（P7） |
| 校准 | 无（P0-P5）→ Platt（P6）→ isotonic + 低分辨率回退（P7） |
| 门禁 | 无 → AUPRC+ECE（P6）→ +LACE（P8）→ +亚组+逐院区+决策曲线（P7） |
| 基线 | NEWS2（P5）→ +LACE（P8） |
| 监控 | 指标计数（P0）→ PSI 漂移告警 + 门禁内再校准（P7）→ PSI+KS 双指标 + 参考样本（P-Scale） |
| 存储 | SQLite 单后端（P0）→ +PostgreSQL 适配器/Redis Streams（P0 附带，P2-P3 加固） |
| 安全 | 演示令牌（P0）→ fail-closed + CSP + 租户隔离（P1）→ 模型验签（P3） |
| 验证 | 无 → 滚动时间切分 + bootstrap CI（P5）→ golden 钉住（P6）→ 患者级防泄漏 + 报告卡 + 规模演练（P-Scale） |

## 5. 未来演化方向

1. ~~**本轮优化（进行中）**：患者级切分防泄漏（P9 前置必修）、自动模型报告卡、漂移 KS 检验补充、2-3 万出院规模演练~~ **已完成（2026-10-12，P-Scale，见 §3）**
2. **P9 真实数据接入**：按 `docs/data-integration-design.md` 执行出院快照 6 字段契约；真实数据下指标才第一次有意义
3. **对标补齐**（源自 GitHub 同类项目调研）：FHIR R4 导入适配（参考 `mimic-fhir`）、多方法漂移检测（参考 `VectorInstitute/cyclops`）、队列生成独立化（参考 `YAIB`）
4. **生产化**：OIDC 接入、Prometheus 抓取与 SLO 告警、受控模型 Registry/灰度、院内数据治理与合规评估、前瞻性静默运行（见 `docs/clinical-pilot-protocol.md`）

## 6. 关键教训（演化中沉淀）

- **serving 纯净是硬约束**：sklearn/numpy/scipy/lightgbm/optuna 零进入 `app/`，树打分因此用纯 Python 路径行走实现
- **门禁规则"不可用即跳过并注明"**：规则缺失不判失败，但必须在 eval.json `rules` 留痕，避免静默放行
- **每个时间戳不跨折**、种子不得永久挡触发写入、阈值手算基准须复核算术——均为执行期真实踩坑后的裁定
- **终审 + 唯一修复轮**：每阶段用全新上下文审查，防止实现者自批
