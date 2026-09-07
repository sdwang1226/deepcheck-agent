"""离线单元测试：不依赖 API Key / 模型 / 索引，用假数据源验证工具层与评分规则。运行：python -m pytest tests/"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from longbridge.openapi import FinancialReportKind  # noqa: F401
except ImportError:  # 未安装 SDK 时用桩模块替代，保证工具层测试可离线运行
    import types

    stub = types.ModuleType("longbridge.openapi")
    stub.FinancialReportKind = SimpleNamespace(IncomeStatement="IS")
    stub.Config = stub.QuoteContext = stub.FundamentalContext = object
    sys.modules["longbridge"] = types.ModuleType("longbridge")
    sys.modules["longbridge.openapi"] = stub

from deepcheck.tools import build_tools, normalize_symbol
from eval import TEST_CASES, score


class FakeQuoteCtx:
    def quote(self, symbols):
        return [SimpleNamespace(last_done=200.0, prev_close=190.0, volume=1_000_000)]


class FakeFundCtx:
    def financial_report(self, symbol, kind):
        return SimpleNamespace(list={"IS": {"indicators": [{"accounts": [
            {"field": "OperatingRevenue", "name": "营业收入", "values": [{"value": "111184000000", "yoy": "16.6"}]},
            {"field": "Other", "name": "无关字段", "values": [{"value": "1"}]},
        ]}]}})


class FakeIndex:
    def similarity_search(self, query, k=2):
        return [SimpleNamespace(page_content=f"chunk about {query}") for _ in range(k)]


REGISTRY = {"AAPL.US": {"name": "Apple Inc."}, "NVDA.US": {"name": "NVIDIA Corporation"}}
quote_tool, fin_tool, rag_tool = build_tools(FakeQuoteCtx(), FakeFundCtx(), {t: FakeIndex() for t in REGISTRY}, REGISTRY)


def test_normalize_symbol():
    assert normalize_symbol("aapl") == "AAPL.US"
    assert normalize_symbol(" 00700.HK ") == "00700.HK"


def test_quote_has_source_prefix_and_change():
    out = quote_tool.invoke({"symbol": "AAPL"})
    assert out.startswith("[来源: Longbridge Quote API]")
    assert "+5.26%" in out


def test_financial_filters_fields():
    out = fin_tool.invoke({"symbol": "AAPL.US"})
    assert "[来源: Longbridge Fundamental API | AAPL.US]" in out
    assert "营业收入" in out and "同比 +16.6%" in out
    assert "无关字段" not in out


def test_rag_routes_by_symbol_and_rejects_unknown():
    out = rag_tool.invoke({"query": "risk factors", "symbol": "nvda"})
    assert out.startswith("[来源: NVIDIA Corporation (NVDA.US) 10-K]")
    assert "片段2" in out
    err = rag_tool.invoke({"query": "risk factors", "symbol": "BABA.US"})
    assert err.startswith("[错误]") and "AAPL.US" in err


def test_rag_symbol_is_required():
    assert rag_tool.args_schema.model_json_schema()["required"] == ["query", "symbol"]


def test_score_full_marks_and_partial():
    tc = TEST_CASES[0]  # 行情
    full = score(tc, "[来源: Longbridge Quote API] AAPL 当前股价 $200.00，较昨收 $190.00 上涨 +5.26%，成交量 1,000,000。", [{"name": "get_stock_quote"}], [])
    assert full["总计"] == 8
    wrong_tool = score(tc, "没有数据", [{"name": "search_10k_report"}], [])
    assert wrong_tool["工具调用"] == 0 and wrong_tool["来源引用"] == 0
    mixed = TEST_CASES[15]  # 混合：quote + 10k
    partial = score(mixed, "x" * 60, [{"name": "get_stock_quote"}], [])
    assert partial["工具调用"] == 1


def test_case_count():
    assert len(TEST_CASES) == 20
