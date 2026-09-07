"""Agent 工具集。所有工具返回值在数据层强制携带 `[来源: ...]` 前缀，引用不依赖 Prompt 约束。"""
import time

from langchain.tools import tool
from longbridge.openapi import FinancialReportKind

FINANCIAL_FIELDS = ["OperatingRevenue", "NetProfit", "EPS", "GrossMgn", "NetProfitMargin", "ROE"]


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    return symbol if "." in symbol else f"{symbol}.US"


def build_tools(quote_ctx, fund_ctx, stock_indices: dict, stock_registry: dict):
    available = ", ".join(stock_indices.keys())

    @tool
    def get_stock_quote(symbol: str) -> str:
        """获取股票实时行情（现价、昨收、涨跌幅、成交量）。symbol 如 AAPL.US 或 00700.HK。"""
        symbol = normalize_symbol(symbol)
        try:
            q = quote_ctx.quote([symbol])[0]
        except Exception as e:
            return f"[错误] 获取 {symbol} 行情失败: {e}"
        change = (q.last_done - q.prev_close) / q.prev_close * 100
        return (
            f"[来源: Longbridge Quote API]\n"
            f"{symbol} ${q.last_done:.2f} | 昨收: ${q.prev_close:.2f} "
            f"| 涨跌: {change:+.2f}% | 量: {q.volume:,}"
        )

    @tool
    def get_financial_data(symbol: str) -> str:
        """获取最新季度核心财务指标（营收、净利润、EPS、毛利率、净利率、ROE）。symbol 如 AAPL.US 或 00700.HK。"""
        symbol = normalize_symbol(symbol)
        time.sleep(1)  # Fundamental API 限频
        try:
            income = fund_ctx.financial_report(symbol, kind=FinancialReportKind.IncomeStatement)
        except Exception as e:
            return f"[错误] 获取 {symbol} 财务数据失败: {e}"
        lines = [f"[来源: Longbridge Fundamental API | {symbol}]"]
        for block in income.list.get("IS", {}).get("indicators", []):
            for acc in block.get("accounts", []):
                if acc.get("field") not in FINANCIAL_FIELDS:
                    continue
                v = acc["values"][0]
                yoy = v.get("yoy", "")
                yoy_str = f"（同比 {float(yoy):+.1f}%）" if yoy else ""
                lines.append(f"  {acc['name']}: {v['value']} {yoy_str}")
        if len(lines) == 1:
            lines.append("  暂无数据")
        return "\n".join(lines)

    @tool
    def search_10k_report(query: str, symbol: str) -> str:
        """在指定股票的 10-K 年报全文中做语义检索（风险因素、业务描述、收入构成、财务政策等）。
        query: 检索内容，建议用英文，如 "risk factors" / "revenue by segment"。
        symbol: 股票代码，必填。若用户和对话历史都没有指明股票，不要猜测，先向用户确认。
        """
        symbol = normalize_symbol(symbol)
        if symbol not in stock_indices:
            return f"[错误] 未找到 {symbol} 的年报索引。当前支持: {available}"
        company = stock_registry[symbol]["name"]
        docs = stock_indices[symbol].similarity_search(query, k=2)
        results = [f"[来源: {company} ({symbol}) 10-K]"]
        for i, doc in enumerate(docs):
            results.append(f"\n--- 片段{i + 1} ---\n{doc.page_content[:400]}")
        return "\n".join(results)

    return [get_stock_quote, get_financial_data, search_10k_report]
