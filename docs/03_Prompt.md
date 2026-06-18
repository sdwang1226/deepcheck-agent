# DeepCheck 投研 Agent — Prompt 迭代记录

> 记录 System Prompt 从 v1 到 v5 的演进过程，包含每次改了什么、为什么改、效果对比。

---

## v1：基础版

### Prompt

```
You are a financial research assistant.
Tools: {tools}
----------------
Question: {input}
Thought: {agent_scratchpad}
```

### 现象

- Agent 能调工具，但**不知道什么时候调用哪个工具**
- 回答中出现了 `$111184000000.0000` 而非 `$1111.8亿`
- 没有来源标注
- DeepSeek 的 ReAct 格式经常输出 `[调用 xxx]` 中文格式 → 解析失败

### 改进方向

→ 加规则约束（v2）、加来源引用（v3）、换 Tool Calling 模式（v4）

---

## v2：加规则约束

### Prompt 改动

```
新增规则:
1. NEVER make up numbers
2. Format: "$111.18B" not "111184000000.0000"
3. Cite data source
```

### 效果

- 数值格式化改善了（但偶尔还是出现原始格式）
- ReAct 格式错误仍然频繁

---

## v3：加来源标注模板

### 改动：不是改 Prompt，是改工具返回值

```python
# 旧版（没有来源）
return f"AAPL ${q.last_done}"

# 新版（强制来源前缀）
return f"[来源: Longbridge Quote API]\nAAPL ${q.last_done}"
```

### 设计思路

> "与其让 LLM 记住'必须标注来源'这条规则，不如在工具返回值层面强制嵌入来源信息。LLM 自然会引用它看到的内容。"

### 效果

- 来源引用率从 ~30% → **90%+**
- 这是整个迭代中 ROI 最高的一次改动——**改的是数据层，不是 Prompt 层**

---

## v4：换 Tool Calling 模式

### 问题

DeepSeek 在 ReAct 模式下频繁出现格式错误：

```
❌ Invalid Format: Missing 'Action:' after 'Thought:'
❌ [调用 get_stock_quote(symbol="AAPL.US")]  ← 中文格式，无法解析
```

### 解法

```python
# 旧: create_react_agent → LLM 写文本 → 正则解析
# 新: create_tool_calling_agent → LLM 输出 JSON → 直接调函数
```

### Prompt 改动

```
旧: 需要详细的 "Thought → Action → Action Input → Observation" 格式说明
新: 不需要格式说明，模型原生支持 function calling
```

### 效果

- Agent 稳定性：50% → **100%**（20/20 评测全通）
- 并行调用：Round 1 同一轮并行调了行情+财报，响应时间减半

---

## v5：中文查询翻译策略

### 问题

RAG 检索 10-K 时，中文查询匹配不到英文文档。

### Prompt 改动

```
旧: "你是金融投研助手。"
新: "使用 search_10k_report 时必须用英文关键词搜索（10-K是英文文档）。
     例如：问'员工数'→搜'number of employees headcount'
     例如：问'供应链'→搜'supply chain manufacturing locations'"
```

### 效果

- 法律风险查询：2/8 → **6/8**（英文 "legal regulatory risks Item 1A" 命中）
- 但员工数、风险因素仍未改善——根因是 embedding 模型容量不够

---

## v6：多股票路由 + Embedding 升级

### 背景

v5 的两个根因问题：
1. **Embedding 能力不足**：MiniLM (384维/纯英文) 对中文查询匹配英文文档效果差
2. **只支持 Apple 一只股票**，无法展示多股票能力

### 架构层改动

```
Embedding:  MiniLM (384维) → BGE-large-zh-v1.5 (1024维)
向量库:    ChromaDB → FAISS（ChromaDB 在 Windows 上崩溃）
索引架构: 单股票 → 按 symbol 分索引 + registry.json
数据规模: 1只股票 302 chunks → 6只美股 3876 chunks
```

### 工具层改动

```python
# 旧: 只支持 Apple
@tool
def search_10k_report(query: str) -> str:
    docs = vectordb.similarity_search(query, k=2)  # 固定单一索引

# 新: 支持 6 只股票按 symbol 路由
@tool
def search_10k_report(query: str, symbol: str = "AAPL.US") -> str:
    vdb = stock_indices[symbol]                     # 按 symbol 路由
    docs = vdb.similarity_search(query, k=2)
```

### Prompt 层改动

```
新增规则:
5. For 10-K annual report searches, you MUST pass the correct stock symbol.
   Available 10-K reports: AAPL.US (Apple), MSFT.US (Microsoft), ...
6. When searching 10-K reports, always pass both `query` and `symbol` parameters.
```

### 效果

- RAG 命中率：50% → **70%**（BGE 升级）
- 股票覆盖：1 只 → **6 只美股 TOP**
- 端到端测试：AAPL 行情 ✅ + NVDA 10-K ✅ + TSLA 10-K ✅

---

## 当前版本（v6）

```python
# system prompt 中动态注入可用股票列表
AVAILABLE_STOCKS = "AAPL.US (Apple), MSFT.US (Microsoft), ..."

prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are a financial research assistant.

RULES:
1. If user does NOT specify a stock, use the SAME stock from conversation history.
2. Cite data source in every answer.
3. NEVER make up numbers. If not available, say so.
4. Format large numbers for readability.
5. Available 10-K reports: {AVAILABLE_STOCKS}. Always pass correct symbol.
6. When searching 10-K, pass both query and symbol parameters."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
```

---

## 迭代总结

| 版本 | 核心改动 | 改动层 | 效果 | 教训 |
|------|---------|--------|------|------|
| v1 | 基础 ReAct | — | 能用但不稳定 | — |
| v2 | 加规则约束 | Prompt | 格式化改善 | Prompt 规则不如数据层防护 |
| v3 | 工具返回值加来源 | **数据层** | 引用率 30%→90% | **改数据层 > 改 Prompt** |
| v4 | ReAct→Tool Calling | **架构层** | 稳定性 50%→100% | 匹配模型的原生能力 |
| v5 | 中文→英文查询翻译 | Prompt | 法律风险 2→6 | Prompt 能部分弥补模型缺陷 |
| **v6** | **BGE升级 + 多股票路由** | **架构+数据+Prompt** | **命中率 70%, 6股** | **根因在模型能力，不在 Prompt** |

### 最核心的一条教训

> **"能改数据层的不要只改 Prompt，能改架构层的不要只改数据层。"**
>
> v3（工具返回值加来源前缀）只改了 3 行代码，效果超过 v2 里的一大段 Prompt 规则。
> v6（换 BGE 模型）解决了 v5 中 Prompt 无法弥补的根因问题——embedding 能力不足。
