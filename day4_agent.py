"""
Day 4: Agent 引用机制 + 对话记忆
=================================
使用方法：
  1. 确保代理已连接（curl -x http://127.0.0.1:7897 https://api.deepseek.com）
  2. 在 Jupyter 中逐个 Cell 执行
  3. 看 verbose 输出观察 Agent 的推理 + 工具调用 + 记忆链路
"""
# %% Cell 1: 初始化
import os, time, warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

# ==== 网络配置 ====
# DeepSeek 走代理，Longbridge 走直连
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['NO_PROXY'] = 'open.longportapp.com,openapi.longbridge.com'

# Patch transformers torch version check (torch 2.4.1, check requires 2.6+)
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

# ==== Longbridge API ====
from longbridge.openapi import (
    Config, QuoteContext, FundamentalContext, FinancialReportKind,
)
config = Config.from_apikey(
    app_key=os.getenv("LONGBRIDGE_APP_KEY"),
    app_secret=os.getenv("LONGBRIDGE_APP_SECRET"),
    access_token=os.getenv("LONGBRIDGE_ACCESS_TOKEN"),
)
quote_ctx = QuoteContext(config)
fund_ctx = FundamentalContext(config)

# ==== RAG 多股票向量库 ====
import json
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

class SE(Embeddings):
    def __init__(self):
        self.m = SentenceTransformer(
            "BAAI/bge-large-zh-v1.5",
            device="cpu", local_files_only=True
        )
    def embed_documents(self, t):
        return self.m.encode(t, normalize_embeddings=True).tolist()
    def embed_query(self, t):
        return self.m.encode([t], normalize_embeddings=True)[0].tolist()

BASE = os.path.dirname(os.path.abspath(__file__))
embedding = SE()

# 加载多股票索引注册表
with open(f"{BASE}/data/faiss_indices/registry.json", "r", encoding="utf-8") as f:
    stock_registry = json.load(f)

# 预加载所有索引
stock_indices = {}
for ticker, info in stock_registry.items():
    stock_indices[ticker] = FAISS.load_local(
        info["index_path"], embedding, allow_dangerous_deserialization=True
    )

AVAILABLE_STOCKS = ", ".join([f"{t} ({info['name']})" for t, info in stock_registry.items()])
print(f"✅ API + RAG 就绪 | {len(stock_indices)} 只股票: {AVAILABLE_STOCKS}")

# %% Cell 2: 定义带来源标注的工具
from langchain.tools import tool

@tool
def get_stock_quote(symbol: str) -> str:
    """获取股票实时行情。输入股票代码如 AAPL.US 或 00700.HK。"""
    if '.' not in symbol:
        symbol = symbol + '.US'
    q = quote_ctx.quote([symbol])[0]
    c = (q.last_done - q.prev_close) / q.prev_close * 100
    return (
        f"[来源: Longbridge Quote API]\n"
        f"{symbol} ${q.last_done:.2f} | 昨收: ${q.prev_close:.2f} "
        f"| 涨跌: {c:+.2f}% | 量: {q.volume:,}"
    )

@tool
def get_financial_data(symbol: str) -> str:
    """获取最新季度核心财务指标（营收、净利润、EPS、毛利率、净利率、ROE）。"""
    if '.' not in symbol:
        symbol = symbol + '.US'
    time.sleep(1)
    income = fund_ctx.financial_report(symbol, kind=FinancialReportKind.IncomeStatement)
    lines = [f"[来源: Longbridge Fundamental API | {symbol}]"]
    for block in income.list.get('IS', {}).get('indicators', []):
        for acc in block.get('accounts', []):
            f = acc.get('field', '')
            if f in ['OperatingRevenue', 'NetProfit', 'EPS',
                     'GrossMgn', 'NetProfitMargin', 'ROE']:
                v = acc['values'][0]
                yoy = v.get('yoy', '')
                ys = f"（同比 {float(yoy):+.1f}%）" if yoy else ""
                lines.append(f"  {acc['name']}: {v['value']} {ys}")
    return '\n'.join(lines)

@tool
def search_10k_report(query: str, symbol: str = "AAPL.US") -> str:
    """搜索股票的 10-K 年报全文，获取风险因素、业务描述、财务政策等。
    参数:
      query: 搜索内容，如"risk factors"或"revenue breakdown"
      symbol: 股票代码，如 AAPL.US, MSFT.US, GOOGL.US, NVDA.US, TSLA.US, AMZN.US
    """
    if '.' not in symbol:
        symbol = symbol + '.US'
    symbol = symbol.upper()
    if symbol not in stock_indices:
        available = ', '.join(stock_indices.keys())
        return f"[错误] 未找到 {symbol} 的年报索引。当前支持: {available}"
    vdb = stock_indices[symbol]
    company = stock_registry[symbol]['name']
    docs = vdb.similarity_search(query, k=2)
    results = [f"[来源: {company} ({symbol}) 10-K]"]
    for i, doc in enumerate(docs):
        results.append(f"\n--- 片段{i+1} ---\n{doc.page_content[:400]}")
    return '\n'.join(results)

tools = [get_stock_quote, get_financial_data, search_10k_report]
print("✅ 工具就绪:", [t.name for t in tools])

# %% Cell 3: 创建 Agent（记忆 + 引用）
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

template = """You are a financial research assistant.

CRITICAL RULES:
1. If user does NOT specify a stock, use the SAME stock from conversation history.
2. Cite data source in every answer. Format like "[来源: Longbridge Quote API]"
3. NEVER make up numbers. If not available, say so explicitly.
4. Format large numbers for readability: "$111.18B" not "111184000000.0000"
5. For 10-K annual report searches, you MUST pass the correct stock symbol. Available 10-K reports: """ + AVAILABLE_STOCKS + """
6. When searching 10-K reports, always pass both `query` and `symbol` parameters.

Tools: {tools}
Tool names: {tool_names}

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input","chat_history","agent_scratchpad","tools","tool_names"],
)

llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0,
)
agent = create_react_agent(llm, tools, prompt)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
executor = AgentExecutor(
    agent=agent, tools=tools, memory=memory,
    verbose=True, handle_parsing_errors=True, max_iterations=5,
)

print("✅ Agent 就绪（带记忆 + 来源引用）")

# %% Cell 4: 4轮对话测试
# 观察要点：
#   Round 2: 没指定股票 → Agent 应自动使用 Round 1 的 AAPL
#   Round 4: 没指定股票 → Agent 应自动使用 Round 3 的 00700

rounds = [
    ("Round 1: 指定 AAPL", "AAPL current price and latest revenue?"),
    ("Round 2: 不指定股票", "Its main risk factors from 10-K?"),
    ("Round 3: 切换到 NVDA", "What are NVIDIA's main risk factors from its 10-K?"),
    ("Round 4: 切换到 TSLA", "Search Tesla 10-K for revenue breakdown"),
    ("Round 5: 切换到腾讯", "Now check 00700.HK Tencent price"),
    ("Round 6: 不指定股票", "Its financial data?"),
]

for label, query in rounds:
    print(f"\n{'='*60}")
    print(label)
    print(f"Query: {query}")
    print(f"{'='*60}")
    r = executor.invoke({"input": query})
    print("\n>>>", r["output"])
