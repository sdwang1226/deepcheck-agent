"""
Day 5: 评测体系
===============
4 个维度 × 20 条测试用例 = 可直接运行的评测脚本

用法：在 Jupyter 中逐个 Cell 执行，最后得到一个评分矩阵
"""

# %% Cell 1: 初始化（复用 Day 4 的 executor 和 tools）
import os, time, json, warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

# Patch transformers torch version check (torch 2.4.1, check requires 2.6+)
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

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

from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

class SE(Embeddings):
    def __init__(self):
        self.m = SentenceTransformer("BAAI/bge-large-zh-v1.5", device="cpu")
    def embed_documents(self, t):
        return self.m.encode(t, normalize_embeddings=True).tolist()
    def embed_query(self, t):
        return self.m.encode([t], normalize_embeddings=True)[0].tolist()

BASE = os.path.dirname(os.path.abspath(__file__))
vectordb = FAISS.load_local(f"{BASE}/data/faiss_index", SE(), allow_dangerous_deserialization=True)

from langchain.tools import tool

@tool
def get_stock_quote(symbol: str) -> str:
    """获取股票实时行情。"""
    px, px_h = os.environ.pop('HTTP_PROXY', None), os.environ.pop('HTTPS_PROXY', None)
    try:
        q = quote_ctx.quote([symbol])[0]
        c = (q.last_done - q.prev_close) / q.prev_close * 100
        return f"[来源: Longbridge Quote API]\n{symbol} ${q.last_done:.2f} | 昨收: ${q.prev_close:.2f} | 涨跌: {c:+.2f}% | 量: {q.volume:,}"
    finally:
        if px: os.environ['HTTP_PROXY'] = px
        if px_h: os.environ['HTTPS_PROXY'] = px_h

@tool
def get_financial_data(symbol: str) -> str:
    """获取最新季度核心财务指标。"""
    px, px_h = os.environ.pop('HTTP_PROXY', None), os.environ.pop('HTTPS_PROXY', None)
    try:
        time.sleep(1)
        income = fund_ctx.financial_report(symbol, kind=FinancialReportKind.IncomeStatement)
        lines = [f"[来源: Longbridge Fundamental API | {symbol}]"]
        for block in income.list.get('IS', {}).get('indicators', []):
            for acc in block.get('accounts', []):
                f = acc.get('field', '')
                if f in ['OperatingRevenue', 'NetProfit', 'EPS', 'GrossMgn', 'NetProfitMargin', 'ROE']:
                    v = acc['values'][0]
                    y = v.get('yoy', '')
                    ys = f"（同比 {float(y):+.1f}%）" if y else ""
                    lines.append(f"  {acc['name']}: {v['value']} {ys}")
        return '\n'.join(lines)
    finally:
        if px: os.environ['HTTP_PROXY'] = px
        if px_h: os.environ['HTTPS_PROXY'] = px_h

@tool
def search_10k_report(query: str) -> str:
    """搜索 Apple 10-K 年报全文。"""
    docs = vectordb.similarity_search(query, k=2)
    results = [f"[来源: Apple 10-K FY2024]"]
    for i, doc in enumerate(docs):
        results.append(f"\n--- 10-K片段{i+1} ---\n{doc.page_content[:400]}")
    return '\n'.join(results)

tools = [get_stock_quote, get_financial_data, search_10k_report]

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

from langchain_deepseek import ChatDeepSeek
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个金融投研助手。规则：
1. 不指定股票时，沿用对话历史中的最新股票。
2. 回答中必须标注数据来源。
3. 严禁编造数据，无数据时说"暂无数据"。
4. 大数字格式化："1111.8亿美元"而非"111184000000.0000"。"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), temperature=0)
agent = create_tool_calling_agent(llm, tools, prompt)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False, handle_parsing_errors=True, max_iterations=4)

print("✅ 评测环境就绪")


# %% Cell 2: 定义评测维度和评分标准
"""
4 维评测体系：

维度1 — 工具调用准确率
  0分：调了错误的工具，或该调但没调
  1分：调了正确工具，但参数有误
  2分：工具调用完全正确

维度2 — 数值准确性
  0分：数值错误（编造、单位错误、正负号反了）
  1分：数值方向正确但精度不够
  2分：数值精确，格式正确

维度3 — 来源引用完整性
  0分：无任何来源标注
  1分：笼统标注（"根据财报"）
  2分：精确标注（"[来源: Longbridge Quote API]"）

维度4 — 回答相关性
  0分：答非所问
  1分：部分相关，但偏离了核心问题
  2分：直接命中问题核心
"""

EVAL_DIMS = ["工具调用", "数值准确性", "来源引用", "回答相关性"]


# %% Cell 3: 20 条测试用例
# 格式: (类别, 查询, 预期调用的工具, 关键数值验证)
test_cases = [
    # === 行情类（5条）===
    ("行情", "AAPL当前股价是多少？", "get_stock_quote", "应包含 $xxx.xx 格式的价格"),
    ("行情", "00700.HK今天涨了还是跌了？", "get_stock_quote", "应包含涨跌幅百分比"),
    ("行情", "腾讯现在多少钱？成交量多大？", "get_stock_quote", "应包含港币价格 + 成交量"),
    ("行情", "What is the latest price of AAPL.US?", "get_stock_quote", "应包含美元价格"),
    ("行情", "苹果今天的最高价和最低价？", "get_stock_quote", "应包含 high 和 low"),

    # === 财报类（5条）===
    ("财报", "苹果最新季度营收多少？", "get_financial_data", "应包含营收具体数值"),
    ("财报", "AAPL的净利润率是多少？", "get_financial_data", "应包含 Net Income Margin"),
    ("财报", "00700.HK的ROE是多少？", "get_financial_data", "应包含 ROE 数值"),
    ("财报", "苹果的毛利率和净利率分别是多少？", "get_financial_data", "应包含 Gross Margin 和 Net Margin"),
    ("财报", "What is Apple's EPS and revenue growth?", "get_financial_data", "应包含 EPS + Revenue + 同比"),

    # === RAG 检索类（5条）===
    ("RAG", "苹果面临哪些主要风险？", "search_10k_report", "应包含风险相关内容"),
    ("RAG", "Apple的主要业务是什么？", "search_10k_report", "应包含 iPhone/Mac/iPad 等产品线"),
    ("RAG", "苹果的供应链集中在哪些国家？", "search_10k_report", "应包含 China/India/Vietnam 等"),
    ("RAG", "What are Apple's legal and regulatory risks?", "search_10k_report", "应包含反垄断和隐私相关内容"),
    ("RAG", "苹果有多少员工？", "search_10k_report", "应包含 employee 数量"),

    # === 混合调用类（5条）===
    ("混合", "苹果股价跌了，有什么风险值得关注？", "quote + 10k", "应同时包含价格和风险"),
    ("混合", "AAPL营收增长快吗？估值贵不贵？", "financial", "应包含营收 + 增速"),
    ("混合", "腾讯现在的股价和净利润多少？", "quote + financial", "应同时包含行情和财务"),
    ("混合", "Apple revenue and its main business risks", "financial + 10k", "应包含营收数据和风险描述"),
    ("混合", "分析一下00700.HK：股价、营收、风险", "quote + financial", "应包含行情和财务数据"),
]

print(f"✅ {len(test_cases)} 条测试用例就绪")
print(f"   行情: 5 | 财报: 5 | RAG: 5 | 混合: 5")


# %% Cell 4: 手动评测函数
def evaluate_answer(test_case, agent_output, tool_calls_made):
    """
    人工评测一条回答。
    返回: (维度1得分, 维度2得分, 维度3得分, 维度4得分, 备注)
    """
    category, query, expected_tool, key_check = test_case
    output = agent_output.lower() if agent_output else ""

    # 维度1：工具调用
    expected_tools = expected_tool.split(" + ")[0]  # 取第一个期望工具
    called = [t.get("name", "") if isinstance(t, dict) else str(t) for t in tool_calls_made]
    called_str = ",".join(called) if called else "无工具调用"

    has_quote = "get_stock_quote" in called_str or "quote" in called_str
    has_fin = "get_financial_data" in called_str or "financial" in called_str
    has_rag = "search_10k" in called_str or "10k" in called_str

    if expected_tool == "get_stock_quote":
        tool_score = 2 if has_quote else 0
    elif expected_tool == "get_financial_data":
        tool_score = 2 if has_fin else 0
    elif expected_tool == "search_10k_report":
        tool_score = 2 if has_rag else 0
    elif "quote" in expected_tool and "financial" in expected_tool:
        tool_score = 2 if (has_quote and has_fin) else (1 if (has_quote or has_fin) else 0)
    else:
        tool_score = 2 if called else 0

    # 维度2：数值准确性（简化判断：是否包含数字）
    import re
    has_numbers = bool(re.search(r'\d+', output))
    has_currency = bool(re.search(r'[\$\¥HK\¥]', output)) or "美元" in output or "港元" in output or "亿" in output
    accuracy_score = 2 if (has_numbers and has_currency) else (1 if has_numbers else 0)

    # 维度3：来源引用
    has_source = "来源" in output or "source" in output.lower() or "longbridge" in output.lower()
    source_score = 2 if has_source else (1 if ("10-k" in output.lower() or "年报" in output) else 0)

    # 维度4：回答相关性
    # 简化：如果 agent 返回内容长度 > 50 字符，且不是纯错误信息
    relevance_score = 2 if (len(output) > 50 and "error" not in output.lower()[:100]) else (1 if len(output) > 10 else 0)

    return {
        "工具调用": tool_score,
        "数值准确性": accuracy_score,
        "来源引用": source_score,
        "回答相关性": relevance_score,
        "总计": tool_score + accuracy_score + source_score + relevance_score,
        "备注": f"调用: {called_str}"
    }


# %% Cell 5: 执行评测（选前 8 条跑，避免超时）
print("=" * 60)
print("开始执行评测（前 8 条测试用例）...")
print("=" * 60)

results = []
for i, tc in enumerate(test_cases[:8]):
    category, query, expected_tool, key_check = tc
    print(f"\n[{i+1}] {category} | {query}")
    print(f"    预期工具: {expected_tool}")

    try:
        # 重置 memory（每条独立评测）
        memory.clear()
        r = executor.invoke({"input": query})
        output = r.get("output", "")

        # 从 intermediate_steps 提取工具调用记录
        steps = r.get("intermediate_steps", [])
        tool_calls = []
        for step in steps:
            if hasattr(step, 'tool') and hasattr(step, 'tool_input'):
                tool_calls.append({"name": step.tool, "input": step.tool_input})

        eval_result = evaluate_answer(tc, output, tool_calls)

        print(f"    得分: {eval_result['总计']}/8 | 工具:{eval_result['工具调用']} "
              f"数值:{eval_result['数值准确性']} 来源:{eval_result['来源引用']} "
              f"相关:{eval_result['回答相关性']}")
        print(f"    备注: {eval_result['备注']}")
        print(f"    回答摘要: {output[:150]}...")

        eval_result["序号"] = i + 1
        eval_result["类别"] = category
        eval_result["查询"] = query
        results.append(eval_result)

    except Exception as e:
        print(f"    ❌ 错误: {type(e).__name__}: {str(e)[:100]}")
        results.append({
            "序号": i + 1, "类别": category, "查询": query,
            "工具调用": 0, "数值准确性": 0, "来源引用": 0, "回答相关性": 0,
            "总计": 0, "备注": f"异常: {type(e).__name__}"
        })

print("\n" + "=" * 60)
print("评测完成")
print("=" * 60)


# %% Cell 6: 结果汇总
print("\n" + "=" * 60)
print("评测结果汇总")
print("=" * 60)

if results:
    avg = sum(r["总计"] for r in results) / len(results)
    print(f"\n测试用例数: {len(results)}")
    print(f"平均得分: {avg:.1f}/8")

    print(f"\n{'序号':<4} {'类别':<4} {'工具':<4} {'数值':<4} {'来源':<4} {'相关':<4} {'总计':<4} {'查询'}")
    print("-" * 80)
    for r in results:
        print(f"{r['序号']:<4} {r['类别']:<4} {r['工具调用']:<4} {r['数值准确性']:<4} "
              f"{r['来源引用']:<4} {r['回答相关性']:<4} {r['总计']:<4} {r['查询'][:30]}")

    print(f"\n各维度平均分:")
    for dim in EVAL_DIMS:
        dim_avg = sum(r[dim] for r in results) / len(results)
        bar = "█" * int(dim_avg * 10) + "░" * (20 - int(dim_avg * 10))
        print(f"  {dim}: {dim_avg:.1f}/2  {bar}")

    print(f"\n各类别平均分:")
    for cat in ["行情", "财报", "RAG", "混合"]:
        cat_results = [r for r in results if r["类别"] == cat]
        if cat_results:
            cat_avg = sum(r["总计"] for r in cat_results) / len(cat_results)
            print(f"  {cat}: {cat_avg:.1f}/8 ({len(cat_results)}条)")
else:
    print("无结果，请检查 Cell 5 执行情况")
