# 售后小蜜 · 电商售后智能客服 Agent

基于 **LangGraph ReAct** 编排的多工具售后客服 Agent，集成 **混合检索 RAG**（查询改写 → BM25+向量双路 → RRF 融合 → Rerank 精排 → 增量索引），并以 **确定性规则引擎**兜底合规判定，杜绝政策幻觉。

> 本项目为学习/求职演示作品，由 RAG+Agent 工程骨架改造而来，业务场景为电商售后客服。

---

## 一、为什么这样设计（面向"大模型应用 / Agent 工程师"）

| 设计点 | 解决的问题 | 面试可讲的点 |
|---|---|---|
| **ReAct 多工具编排** | 售后问题需要跨订单/物流/退款/政策多个系统取证 | 工具自主选择的决策链路，事件流可观测 |
| **规则引擎兜底合规** | LLM 算日期、判责任方易出错，涉合规风险 | "LLM 做语义理解，规则引擎做合规判定" |
| **混合检索 RAG** | 单纯向量检索对政策条文召回不准 | 改写→双路检索→RRF→Rerank→增量索引全链路 |
| **上下文瘦身** | 多轮对话+大段政策撑爆上下文 | 历史去冗余 + 实体注入 |
| **动态场景路由** | 普通问答与工单生成需要不同人设 | 中间件运行时切换 system prompt |

**核心亮点：Agent 不自己算合规。** 是否超七天无理由窗口、运费谁承担、是否保修，全部由 `agent/tools/return_rules.py` 的纯函数判定（严格遵循"签收次日起算"法定口径），LLM 只负责理解语义并解释结论。

---

## 二、数据来源说明（重要）

| 数据 | 来源 | 性质 |
|---|---|---|
| `data/policy_*.txt` | 《网络购买商品七日无理由退货暂行办法》（国家市场监督管理总局令第31号，[gov.cn 原文](https://www.gov.cn/zhengce/zhengceku/2020-11/03/content_5557118.htm)）、《消费者权益保护法》退货条款 | **真实法规** |
| `data/shop_01_演示店铺售后总则.txt` | 基于主流电商平台公开的售后政策归纳整理 | 归纳整理 |
| `data/shop_02_商品售后FAQ.txt` | 演示用商品问答 | 演示数据 |
| `data/external/*.csv` | 自行构造的订单/物流/退款记录 | **模拟数据**（无真实系统权限） |

> 订单、物流、退款数据均为模拟数据，用于演示 Agent 的工具调用链路；如需接入真实系统，只需替换 `agent/tools/agent_tools.py` 中的数据读取实现为 API 客户端，工具函数签名与 Agent 编排无需改动。

---

## 三、模型接入（重要：先读这段）

本应用支持两种模型接入方式，**Key 永不进入代码/仓库/日志**：

### 方式 A：本地运行，配置自己的 Key（最简单）

```bash
# 1. 创建虚拟环境并安装依赖（Python 3.10+，本项目实测 3.14 可用）
python -m venv .venv
.venv\Scripts\activate        # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置密钥（阿里云百炼 DashScope）
cp .env.example .env
# 编辑 .env 填入 DASHSCOPE_API_KEY（仅本机文件，已被 .gitignore 排除）

# 3. 构建知识库向量索引（首次需要；已 clone 本仓库自带索引可跳过）
python -m rag.vector_store

# 4. 启动应用
streamlit run agent_app.py
```

### 方式 B：在线 Demo / 让别人体验 —— 访问者自带 Key（推荐）

部署到公网后，访问者在左侧 **⚙️ 模型接入** 面板**填写自己的** DashScope / OpenAI 兼容 API Key 即可使用：

- Key 仅保存在**当前浏览器会话**中（`st.session_state`），服务器不存储、不写日志；
- 多人访问**各用各的 Key**，互不串用，不会消耗部署者额度；
- 未填写 Key 前页面只展示项目介绍，不发起任何模型调用；
- 若部署者在 `.env` / 云端 Secrets 中配置了平台 Key，也**不会自动启用**——需访问者显式点击"改用平台预置 Key"（见下）。

> 因此，如果你想开放一个免费 Demo 给面试官体验，**不需要**把任何 Key 放进云端 Secrets——访问者用自己的 Key 即可。

### 快速连通性自检（CLI，可选）

```bash
python scripts/check_api.py      # 测试 chat / embedding / rerank
python scripts/test_rag.py       # 测试政策 RAG 问答
python scripts/test_agent.py     # 测试 Agent 工具调用链路
```

> 提示：模拟业务数据中的"签收日期"固定为 2026 年附近的时间点，RAG 检索、规则引擎判定均以运行当天的 `date.today()` 计算。若你今天提问"能否无理由退货"，结果会随实际日期推移而变化（这正是规则引擎而非 LLM 心算的原因）。如需稳定演示效果，可自行修改 `data/external/orders.csv` 中的签收日期。

---

## 四、免费在线部署（Streamlit Community Cloud）

想让面试官/朋友**点链接直接体验**，可用 [Streamlit Community Cloud](https://share.streamlit.io/)（免费，需 GitHub 账号）。

### 步骤

1. **把本项目推送到你的 GitHub 仓库**（代码内已包含预构建向量索引 `faiss_db/*.pkl`，无需在云端重建）。
2. 打开 [share.streamlit.io](https://share.streamlit.io/) → 用 GitHub 账号登录 → **New app**。
3. 选择仓库 / 分支(`main`) / 入口文件 `agent_app.py` → **Deploy**。
4. 首次部署会自动 `pip install -r requirements.txt`，约 3~5 分钟。
5. 部署完成后即可分享链接。访问者打开后，在侧栏**填入自己的 API Key** 即可使用。

### 部署者要不要配 Secrets？（关键）

**推荐：不要配。** 访问者自带 Key 模式已覆盖绝大多数场景：

| 场景 | 做法 |
|---|---|
| 面试官体验 Demo | 部署者**不配** Secrets，访问者填自己的 Key，零额度风险 |
| 仅自己/团队私有使用 | 可在 Settings → Secrets 填 `DASHSCOPE_API_KEY`，访问者点击"改用平台预置 Key"启用 |
| 完全离线体验 | 不需要（本应用依赖在线 LLM API） |

即使配了平台 Key，也需要访问者**显式点击**才会启用——应用不会静默消耗部署者额度。

### 免费版注意事项

- 免费版闲置一段时间会休眠，首次访问需等待冷启动（约 1~3 分钟）。
- 免费版对同一账号的总资源有配额，适合面试展示，不适合高并发生产使用。
- 无论哪种模式，Key 都不出现在代码或仓库中；访问者 Key 仅存其浏览器会话。

### 云端与本地差异说明

| 项目 | 说明 |
|---|---|
| 向量索引 | 已随仓库提交 `faiss_db/{faiss_index,bm25_index,file_hashes}.pkl/yml`，云端直接加载，无需重建 |
| 文件系统 | 云端文件系统临时（重启即恢复为 git 快照），`data/` 变更后无法持久化重建索引 |
| 会话记忆 | 云端重启会清空会话历史（Streamlit 本身无持久化），属预期行为 |
| 新增文档 | 生产场景应把知识库变更做成"云端重建任务"或接入数据库；Demo 场景直接改仓库内 `data/` 后重新推送即可 |

---

## 五、演示脚本（按顺序提问，效果最佳）

当前演示数据设定"今天"为 2026-09-02，登录用户 `1001`。

| # | 提问 | 展示能力 |
|---|---|---|
| 1 | `耳机进水了能保修吗？` | 政策 RAG 检索 + 引用来源 |
| 2 | `七天无理由的七天从哪天开始算？` | 法规原文精准召回 |
| 3 | `我 8 月 28 号签收的耳机还能退吗？` | 多步工具编排 + 规则引擎判定 + **主动纠错**（发现 8-28 签收的是手表非耳机） |
| 4 | `我的退款为什么被驳回了？订单号 DX20260720001` | 退款查询 + 解读驳回原因 |
| 5 | `订单 DX20260830004 到哪了？怎么还没到` | 物流轨迹查询 + 异常节点识别 |
| 6 | `我要投诉，物流太慢了，给我升级处理` | 触发工单模式，输出结构化工单 + 安抚话术 |

---

## 六、项目结构

```
.
├─ agent_app.py              # Streamlit 入口
├─ agent/
│  ├─ react_agent.py         # LangGraph ReAct Agent 编排
│  ├─ chat_service.py        # 事件流服务层（按会话缓存 Agent，多 Key 隔离）
│  └─ tools/
│     ├─ agent_tools.py      # 售后工具集（订单/物流/退款/政策/工单）
│     ├─ return_rules.py     # ★ 合规规则引擎（纯函数，可单测）
│     └─ middleware.py       # 工具监控 + 日志脱敏 + 工单模式动态切换 prompt
├─ rag/                      # 检索链路
│  ├─ rag_service.py         # 改写→混合检索→RRF→Rerank→上下文组装
│  ├─ vector_store.py        # FAISS + 增量索引
│  ├─ bm25_index.py          # BM25 稀疏检索（jieba 分词）
│  ├─ reranker.py            # qwen3-rerank 精排
│  └─ file_hash_tracker.py   # MD5 增量索引（相对路径 key，跨机器可移植）
├─ model/
│  ├─ runtime_config.py      # ★ 会话级模型配置（访问者 Key / 平台 Key 双模式）
│  └─ factory.py             # 按会话配置动态构建 Chat/Embedding
├─ prompts/                  # 提示词（客服人设/查询理解/政策QA/工单生成）
├─ data/                     # 知识库文档 + 模拟业务数据
├─ faiss_db/                 # 预构建向量索引（已入库，云端直接加载）
├─ ui/
│  ├─ model_panel.py         # ★ 模型接入面板（自填 Key + 连接测试）
│  ├─ render.py              # 渲染层
│  └─ session.py             # 会话状态
└─ scripts/                  # 连通性与冒烟测试脚本
```

---

## 七、安全设计说明

1. **Key 零存储**：API Key 只存在于浏览器 `session_state`（访问者模式）或部署者环境变量（平台模式），代码仓库与日志中绝无明文 Key。
2. **日志脱敏**：`middleware.py` 对工具入参做递归脱敏，敏感字段（api_key/token/secret/…）一律替换为 `***REDACTED***`。
3. **多用户隔离**：Agent/RAG 组件按会话缓存，配置指纹变化即重建；每个访问者只用自己的 Key。
4. **平台 Key 不自动启用**：即使部署者配置了 `.env`/Secrets，也需要访问者显式点击"改用平台预置 Key"，防止公开 Demo 被静默刷额度。
5. **gitignore 防护**：`.env`、`*.log`、`logs/` 均不入库；提交前可用 `git ls-files | grep -E "\.env$|\.log$"` 自查。

---

## 九、RAG 检索链路

```
用户问题
 → 查询改写（LLM 生成规范化检索用语）
 → 双路检索：BM25(关键词，jieba) ∥ FAISS(稠密向量)
 → RRF 融合（bm25_weight 可配，rrf_k=60）
 → 候选池 max(k*3, 30) → qwen3-rerank 精排
 → 头尾交替排列（缓解 Lost-in-middle）
 → 组装上下文（含来源标注）→ LLM 作答并附引用
```

知识库变更时通过 `file_hash_tracker` 比对文件 MD5，仅重建变化的文档分片，并同步重建 BM25 索引。

---

## 十、后续可扩展

- [ ] 接入真实订单/物流/退款 API（替换 CSV 读取层即可）
- [ ] 多轮会话持久化（SQLite）+ 用户实体槽位记忆
- [ ] 工具调用次数/超时上限，超限自动收口转人工
- [ ] 评测集：意图识别准确率、实体抽取 F1、政策问答命中率
- [ ] Docker 部署 + pytest 单测覆盖规则引擎

---

## 十一、License

MIT
