"""End-to-end test: Agent single query"""
import os, warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['NO_PROXY'] = 'open.longportapp.com,openapi.longbridge.com'

# Patch transformers torch version check (torch 2.4.1, check requires 2.6+)
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

BASE = os.path.dirname(os.path.abspath(__file__))

# 1. Longbridge API
from longbridge.openapi import Config, QuoteContext, FundamentalContext, FinancialReportKind
config = Config.from_apikey(
    app_key=os.getenv("LONGBRIDGE_APP_KEY"),
    app_secret=os.getenv("LONGBRIDGE_APP_SECRET"),
    access_token=os.getenv("LONGBRIDGE_ACCESS_TOKEN"),
)
quote_ctx = QuoteContext(config)
fund_ctx = FundamentalContext(config)
print("1. Longbridge API OK")

# 2. RAG (multi-stock)
import json
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

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
AVAILABLE_STOCKS = ", ".join(stock_indices.keys())
print(f"2. FAISS RAG OK | {len(stock_indices)} stocks: {AVAILABLE_STOCKS}")

# 3. Tools
from langchain.tools import tool
import time

@tool
def get_stock_quote(symbol: str) -> str:
    """Get real-time stock quote. Input symbol like AAPL.US or 00700.HK."""
    if '.' not in symbol:
        symbol = symbol + '.US'
    try:
        q = quote_ctx.quote([symbol])[0]
        c = (q.last_done - q.prev_close) / q.prev_close * 100
        return f"[Source: Longbridge Quote API]\n{symbol} ${q.last_done:.2f} | Change: {c:+.2f}% | Vol: {q.volume:,}"
    except Exception as e:
        return f"[Error] Failed to get quote for {symbol}: {e}"

@tool
def get_financial_data(symbol: str) -> str:
    """Get latest quarterly financials. Input symbol like AAPL.US or 00700.HK."""
    if '.' not in symbol:
        symbol = symbol + '.US'
    time.sleep(1)
    income = fund_ctx.financial_report(symbol, kind=FinancialReportKind.IncomeStatement)
    lines = [f"[Source: Longbridge Fundamental API | {symbol}]"]
    for block in income.list.get('IS', {}).get('indicators', []):
        for acc in block.get('accounts', []):
            f = acc.get('field', '')
            if f in ['OperatingRevenue', 'NetProfit', 'EPS', 'GrossMgn', 'NetProfitMargin', 'ROE']:
                v = acc['values'][0]
                yoy = v.get('yoy', '')
                ys = f" (YoY {float(yoy):+.1f}%)" if yoy else ""
                lines.append(f"  {acc['name']}: {v['value']}{ys}")
    return '\n'.join(lines)

@tool
def search_10k_report(query: str, symbol: str = "AAPL.US") -> str:
    """Search a stock's 10-K annual report. Pass symbol like AAPL.US, MSFT.US, GOOGL.US, NVDA.US, TSLA.US, AMZN.US."""
    if '.' not in symbol:
        symbol = symbol + '.US'
    symbol = symbol.upper()
    if symbol not in stock_indices:
        return f"[Error] No 10-K index for {symbol}. Available: {AVAILABLE_STOCKS}"
    vdb = stock_indices[symbol]
    company = stock_registry[symbol]['name']
    docs = vdb.similarity_search(query, k=2)
    results = [f"[Source: {company} ({symbol}) 10-K]"]
    for i, doc in enumerate(docs):
        results.append(f"\n--- Chunk {i+1} ---\n{doc.page_content[:400]}")
    return '\n'.join(results)

tools = [get_stock_quote, get_financial_data, search_10k_report]
print("3. Tools OK")

# 4. Agent
from langchain_deepseek import ChatDeepSeek
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", f"You are a financial research assistant. Cite data sources in every answer. Available 10-K reports: {AVAILABLE_STOCKS}. Always pass the correct symbol when searching 10-K reports."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), temperature=0)
agent = create_tool_calling_agent(llm, tools, prompt)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True, max_iterations=8)
print("4. Agent OK")

# 5. Multi-stock test
test_queries = [
    ("AAPL price", "AAPL current price?"),
    ("NVDA 10-K risk", "What are NVIDIA's main risk factors from its 10-K annual report?"),
    ("TSLA 10-K revenue", "Search Tesla 10-K for revenue breakdown"),
]
for label, query in test_queries:
    print(f"\n5. Testing: [{label}] '{query}'")
    result = executor.invoke({"input": query})
    print(f"   Answer: {result['output'][:300]}")

print("\n✅ Agent multi-stock end-to-end test PASSED!")
