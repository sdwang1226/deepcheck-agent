# DeepCheck — AI 投研尽调 Agent

> 基于 LLM + RAG + Tool Calling 的金融投研助手，通过自然语言对话自动聚合实时行情、财务数据与年报语义检索，输出带来源引用的投研分析。

---

## 核心指标

| 指标 | 数值 |
|------|------|
| 工具路由准确率 | **100%**（20/20 评测全通过） |
| RAG 语义命中率 | **70%**（BGE-large-zh 升级后，较 MiniLM 提升 40%） |
| 来源引用率 | **90%+**（工具返回值层强制标注） |
| 年报覆盖 | **6 只美股 TOP**（AAPL / MSFT / GOOGL / NVDA / TSLA / AMZN） |
| 向量库总量 | **3,876 chunks**（按 symbol 分索引路由） |
| Agent 稳定性 | **100%**（ReAct → Tool Calling 架构升级后） |

---

## 系统架构

```
                        ┌─────────────────────────────┐
                        │         用户输入              │
                        │  "NVIDIA 有什么风险？"        │
                        └─────────────┬───────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────┐
│                      Agent 编排层                              │
│                                                                │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │ DeepSeek LLM  │───▶│  Tool Calling │───▶│ Conversation   │  │
│  │ (deepseek-chat│    │  Agent        │    │ BufferMemory   │  │
│  │  temp=0)      │    │ (函数调用模式) │    │ (多轮记忆)     │  │
│  └──────────────┘    └──────┬───────┘    └────────────────┘  │
│                             │                                  │
│                    ┌────────┼────────┐                        │
│                    ▼        ▼        ▼                        │
│              ┌────────┐┌────────┐┌──────────────┐            │
│              │ 行情   ││ 财报   ││ 年报检索      │            │
│              │ Tool   ││ Tool   ││ Tool(query,  │            │
│              │        ││        ││      symbol) │            │
│              └───┬────┘└───┬────┘└──────┬───────┘            │
└──────────────────┼─────────┼────────────┼────────────────────┘
                   │         │            │
     ┌─────────────┼─────────┼────────────┼────────────────┐
     │             ▼         ▼            ▼                │
     │  ┌──────────────────┐  ┌────────────────────────┐   │
     │  │ Longbridge API    │  │ FAISS 多股票索引         │   │
     │  │ • QuoteContext    │  │ • BGE-large-zh (1024d) │   │
     │  │ • FundamentalCtx  │  │ • 6只美股 3876 chunks  │   │
     │  │                   │  │ • 按 symbol 路由检索    │   │
     │  └──────────────────┘  └────────────────────────┘   │
     └─────────────────────────────────────────────────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  LLM 综合回答     │
          │  + 来源引用       │
          │  + 数值格式化     │
          └─────────────────┘
```

---

## 技术栈

| 层级 | 技术选型 | 说明 |
|------|---------|------|
| LLM | DeepSeek Chat | Tool Calling 模式，temperature=0 |
| Agent 框架 | LangChain | create_tool_calling_agent + AgentExecutor |
| Embedding | BAAI/bge-large-zh-v1.5 | 1024 维，中英双语，本地推理 |
| 向量库 | FAISS | 按 symbol 分索引路由，支持多股票 |
| 数据源 | Longbridge OpenAPI | 实时行情 + 季度财报 |
| 年报来源 | SEC EDGAR | 自动下载 10-K → 清洗 → 分块 → 建索引 |
| 记忆 | ConversationBufferMemory | 多轮对话上下文保持 |

---

## 功能演示

### 场景 1：实时行情 + 财报查询
```
用户: AAPL current price and latest revenue?
Agent: → 调用 get_stock_quote(AAPL.US) + get_financial_data(AAPL.US)
      → [来源: Longbridge Quote API] AAPL $295.95 | -1.10% | Vol: 42.7M
      → [来源: Longbridge Fundamental API] 营收 $111.18B (+16.6% YoY)
```

### 场景 2：多股票年报 RAG 检索
```
用户: What are NVIDIA's main risk factors from its 10-K?
Agent: → 调用 search_10k_report(query="risk factors", symbol="NVDA.US")
      → [来源: NVIDIA Corporation (NVDA.US) 10-K]
      → 汇总 5 类风险：供应链/出口管制/技术竞争/监管/股价
```

### 场景 3：多轮记忆 — 不指定股票自动沿用
```
用户: (Round 1) AAPL current price?        → Agent 查 AAPL
用户: (Round 2) Its main risk factors?     → Agent 自动沿用 AAPL，搜索年报
用户: (Round 3) Now check NVDA price       → Agent 切换到 NVDA
用户: (Round 4) Its revenue breakdown?     → Agent 自动沿用 NVDA，搜索年报
```

---

## 项目结构

```
deepcheck-agent/
├── day4_agent.py              # Agent 完整代码（工具定义 + Prompt + 多轮对话）
├── day5_eval.py               # 评测体系（4 维度 × 20 条用例）
├── test_agent.py              # 多股票端到端自动化测试
├── download_10k.py            # SEC EDGAR 10-K 自动下载 + 建索引
├── rebuild_vectordb.py        # FAISS 向量库重建脚本
├── compare_embeddings.py      # MiniLM vs BGE 命中率对比实验
├── compare_hard.py            # 高难度语义匹配对比实验
├── apple_research.py          # Longbridge API 探索脚本
├── demo.py                    # 面向演示的交互脚本
├── requirements.txt
├── .env.example
├── data/
│   ├── faiss_indices/         # 多股票 FAISS 索引
│   │   ├── registry.json      # 索引注册表
│   │   ├── AAPL_US/           # Apple (302 chunks)
│   │   ├── MSFT_US/           # Microsoft (737 chunks)
│   │   ├── GOOGL_US/          # Alphabet (697 chunks)
│   │   ├── NVDA_US/           # NVIDIA (689 chunks)
│   │   ├── TSLA_US/           # Tesla (836 chunks)
│   │   └── AMZN_US/           # Amazon (615 chunks)
│   ├── faiss_index/           # Apple 单股票索引（兼容旧版）
│   └── 10k_filings/           # 10-K 原文（清洗后纯文本）
└── docs/
    ├── 01_PRD_Agent.md        # 产品需求文档
    ├── 02_架构设计.md          # 系统架构 + 技术决策
    └── 03_Prompt.md           # Prompt v1→v6 迭代记录
```

---

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入 Longbridge API Key 和 DeepSeek API Key
```

### 2. 构建多股票索引（首次运行）

```bash
python download_10k.py
# 自动从 SEC EDGAR 下载 6 只美股 10-K → 清洗 → 分块 → 建 FAISS 索引
```

### 3. 运行 Agent

```bash
python day4_agent.py
# 多轮对话测试：行情查询 → 年报检索 → 股票切换 → 记忆验证
```

### 4. 运行评测

```bash
python day5_eval.py
# 4 维度 × 20 条用例自动评测
```

### 5. 端到端测试

```bash
python test_agent.py
# 自动验证：AAPL 行情 + NVDA 10-K 风险 + TSLA 10-K 收入
```

---

## 关键技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| Agent 模式 | Tool Calling（非 ReAct） | DeepSeek 对 ReAct 格式遵循差，Tool Calling 稳定性 50%→100% |
| 抗幻觉策略 | 工具返回值层强制来源前缀 | 比 Prompt 规则更可靠，引用率 30%→90%（仅改 3 行代码） |
| Embedding | BGE-large-zh-v1.5 (1024d) | 中英双语，命中率 50%→70%（对比 MiniLM 384d） |
| 向量库 | FAISS（非 ChromaDB） | ChromaDB 在 Windows 上 Rust binary 崩溃，FAISS 纯 Python 稳定 |
| 索引架构 | 按 symbol 分索引 + registry.json | 支持多股票路由，新增股票只需追加索引 |

---

## Prompt 迭代历程

| 版本 | 核心改动 | 改动层 | 效果 |
|------|---------|--------|------|
| v1 | 基础 ReAct | — | 能用但不稳定 |
| v2 | 加规则约束 | Prompt | 格式化改善 |
| v3 | 工具返回值加来源 | **数据层** | 引用率 30%→90% |
| v4 | ReAct→Tool Calling | **架构层** | 稳定性 50%→100% |
| v5 | 中文→英文查询翻译 | Prompt | 法律风险查询 2/8→6/8 |
| v6 | 多股票路由 + 可用股票列表注入 | Prompt+架构 | 支持 6 只股票按 symbol 检索 |

> **核心教训："能改数据层的不要只改 Prompt，能改架构层的不要只改数据层。"**

---

## 评测体系

4 维度 × 20 条用例：

| 维度 | 定义 | 得分 |
|------|------|------|
| 工具路由准确率 | Agent 是否调用了正确的工具 | 20/20 |
| 数值准确率 | 回答数值与 API 返回值一致 | ~80% |
| 来源引用率 | 回答中是否标注了来源 | 90%+ |
| RAG 语义命中率 | 检索结果是否包含答案 | 70% |

---

## 设计文档

- [产品需求文档 (PRD)](docs/01_PRD_Agent.md) — 产品定位、用户场景、MVP 范围、KPI、竞品分析
- [架构设计文档](docs/02_架构设计.md) — 系统架构图、技术决策、RAG 选型、多股票索引设计
- [Prompt 迭代记录](docs/03_Prompt.md) — v1→v6 六版 Prompt 演进，含效果对比与教训总结

---

## License

MIT
