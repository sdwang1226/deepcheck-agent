"""
DeepCheck Agent — 交互式演示脚本
================================
用法：python demo.py
录屏建议：2 分钟内展示 3 个核心场景
"""
import os, time, warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['NO_PROXY'] = 'open.longportapp.com,openapi.longbridge.com'

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

# ── 初始化 ──────────────────────────────────────────────
import json
from longbridge.openapi import Config, QuoteContext, FundamentalContext, FinancialReportKind
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

BASE = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("  DeepCheck — AI 投研尽调 Agent")
print("  Tech: LLM + RAG + Tool Calling + Multi-Stock Index")
print("=" * 60)
print("\n[*] 加载模型与索引...")

config = Config.from_apikey(
    app_key=os.getenv("LONGBRIDGE_APP_KEY"),
    app_secret=os.getenv("LONGBRIDGE_APP_SECRET"),
    access_token=os.getenv("LONGBRIDGE_ACCESS_TOKEN"),
)
quote_ctx = QuoteContext(config)
fund_ctx = FundamentalContext(config)

class SE(Embeddings):
    def __init__(self):
        self.m = SentenceTransformer("BAAI/bge-large-zh-v1.5", device="cpu", local_files_only=True)
    def embed_documents(self, t):
        return self.m.encode(t, normalize_embeddings=True).tolist()
    def embed_query(self, t):
        return self.m.encode([t], normalize_embeddings=True)[0].tolist()

embedding = SE()
with open(f"{BASE}/data/faiss_indices/registry.json", "r", encoding="utf-8") as f:
    stock_registry = json.load(f)
stock_indices = {}
for ticker, info in stock_registry.items():
    stock_indices[ticker] = FAISS.load_local(info["index_path"], embedding, allow_dangerous_deserialization=True)
AVAILABLE = ", ".join([f"{t} ({info['name']})" for t, info in stock_registry.items()])
print(f"[OK] 就绪 | {len(stock_indices)} 只股票: {AVAILABLE}\n")

# ── 工具定义 ────────────────────────────────────────────
@tool
def get_stock_quote(symbol: str) -> str:
    """获取股票实时行情。输入股票代码如 AAPL.US 或 00700.HK。"""
    if '.' not in symbol: symbol = symbol + '.US'
    q = quote_ctx.quote([symbol])[0]
    c = (q.last_done - q.prev_close) / q.prev_close * 100
    return (f"[来源: Longbridge Quote API]\n"
            f"{symbol} ${q.last_done:.2f} | 昨收: ${q.prev_close:.2f} "
            f"| 涨跌: {c:+.2f}% | 量: {q.volume:,}")

@tool
def get_financial_data(symbol: str) -> str:
    """获取最新季度核心财务指标（营收、净利润、EPS、毛利率、净利率、ROE）。"""
    if '.' not in symbol: symbol = symbol + '.US'
    time.sleep(1)
    income = fund_ctx.financial_report(symbol, kind=FinancialReportKind.IncomeStatement)
    lines = [f"[来源: Longbridge Fundamental API | {symbol}]"]
    for block in income.list.get('IS', {}).get('indicators', []):
        for acc in block.get('accounts', []):
            f = acc.get('field', '')
            if f in ['OperatingRevenue', 'NetProfit', 'EPS', 'GrossMgn', 'NetProfitMargin', 'ROE']:
                v = acc['values'][0]
                yoy = v.get('yoy', '')
                ys = f"（同比 {float(yoy):+.1f}%）" if yoy else ""
                lines.append(f"  {acc['name']}: {v['value']} {ys}")
    return '\n'.join(lines)

@tool
def search_10k_report(query: str, symbol: str = "AAPL.US") -> str:
    """搜索股票的 10-K 年报全文，获取风险因素、业务描述、财务政策等。
    参数: query=搜索内容, symbol=股票代码(AAPL.US/MSFT.US/GOOGL.US/NVDA.US/TSLA.US/AMZN.US)"""
    if '.' not in symbol: symbol = symbol + '.US'
    symbol = symbol.upper()
    if symbol not in stock_indices:
        return f"[错误] 未找到 {symbol} 的年报索引。当前支持: {', '.join(stock_indices.keys())}"
    vdb = stock_indices[symbol]
    company = stock_registry[symbol]['name']
    docs = vdb.similarity_search(query, k=2)
    results = [f"[来源: {company} ({symbol}) 10-K]"]
    for i, doc in enumerate(docs):
        results.append(f"\n--- 片段{i+1} ---\n{doc.page_content[:400]}")
    return '\n'.join(results)

tools = [get_stock_quote, get_financial_data, search_10k_report]

# ── Agent 创建 ──────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are a financial research assistant (DeepCheck Agent).

RULES:
1. If user does NOT specify a stock, use the SAME stock from conversation history.
2. Cite data source in every answer.
3. NEVER make up numbers. If not available, say so.
4. Format large numbers for readability.
5. Available 10-K reports: {AVAILABLE}. Always pass correct symbol.
6. When searching 10-K, pass both query and symbol parameters."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), temperature=0)
agent = create_tool_calling_agent(llm, tools, prompt)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False, handle_parsing_errors=True, max_iterations=8)

# ── 交互循环 ────────────────────────────────────────────
print("-" * 60)
print(">> 输入问题开始对话 (输入 q 退出, 输入 demo 运行预设演示)")
print("-" * 60)

DEMO_QUERIES = [
    ("[1] 实时行情", "AAPL current price?"),
    ("[2] NVIDIA 年报风险", "What are NVIDIA's main risk factors from its 10-K annual report?"),
    ("[3] 多轮记忆", "Its latest revenue and EPS?"),
    ("[4] Tesla 年报检索", "Search Tesla 10-K for revenue breakdown by segment"),
]

while True:
    try:
        user_input = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if not user_input:
        continue
    if user_input.lower() == 'q':
        print("\nBye!")
        break

    if user_input.lower() == 'demo':
        for label, query in DEMO_QUERIES:
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"  Query: {query}")
            print(f"{'='*60}")
            result = executor.invoke({"input": query})
            print(f"\nAgent:\n{result['output']}")
        print(f"\n{'='*60}")
        print("  [OK] Demo 完成!")
        print(f"{'='*60}")
        continue

    print("[thinking...]")
    result = executor.invoke({"input": user_input})
    print(f"\nAgent:\n{result['output']}")
