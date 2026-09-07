"""
评测体系：4 维度 × 20 条用例，自动打分并落盘。
================================================
用法：
  python eval.py              # 跑全部 20 条
  python eval.py --limit 8    # 只跑前 8 条（快速回归）
结果写入 eval_results/<时间戳>.json，含每条用例的工具调用、得分与回答摘要。

评分标准（每维 0/1/2 分）：
  工具调用   0 调错/没调 | 1 工具对参数错 | 2 完全正确
  数值准确性 0 无数值/编造 | 1 有数值无单位 | 2 有数值且带货币/单位（自动规则为代理指标，精确核对需人工比对 API 原值）
  来源引用   0 无来源 | 1 笼统（"根据年报"） | 2 精确（"[来源: Longbridge Quote API]"）
  回答相关性 0 答非所问 | 1 部分相关 | 2 直接命中
"""
import json
import os
import re
import sys
import time
from datetime import datetime

from deepcheck import config
from deepcheck.agent import build_agent

EVAL_DIMS = ["工具调用", "数值准确性", "来源引用", "回答相关性"]

# (类别, 查询, 期望工具集合, 关键验证点)
TEST_CASES = [
    # === 行情类（5条）===
    ("行情", "AAPL当前股价是多少？", {"get_stock_quote"}, "应包含 $xxx.xx 格式的价格"),
    ("行情", "00700.HK今天涨了还是跌了？", {"get_stock_quote"}, "应包含涨跌幅百分比"),
    ("行情", "腾讯现在多少钱？成交量多大？", {"get_stock_quote"}, "应包含港币价格 + 成交量"),
    ("行情", "What is the latest price of AAPL.US?", {"get_stock_quote"}, "应包含美元价格"),
    ("行情", "英伟达现在股价多少？", {"get_stock_quote"}, "应包含美元价格"),
    # === 财报类（5条）===
    ("财报", "苹果最新季度营收多少？", {"get_financial_data"}, "应包含营收具体数值"),
    ("财报", "AAPL的净利润率是多少？", {"get_financial_data"}, "应包含净利率"),
    ("财报", "00700.HK的ROE是多少？", {"get_financial_data"}, "应包含 ROE 数值"),
    ("财报", "苹果的毛利率和净利率分别是多少？", {"get_financial_data"}, "应包含毛利率和净利率"),
    ("财报", "What is Apple's EPS and revenue growth?", {"get_financial_data"}, "应包含 EPS + Revenue + 同比"),
    # === RAG 检索类（5条，覆盖多股票路由）===
    ("RAG", "苹果面临哪些主要风险？", {"search_10k_report"}, "应包含风险相关内容"),
    ("RAG", "Apple的主要业务是什么？", {"search_10k_report"}, "应包含 iPhone/Mac/iPad 等产品线"),
    ("RAG", "英伟达年报里提到的出口管制风险是什么？", {"search_10k_report"}, "应路由到 NVDA.US 索引"),
    ("RAG", "What are Tesla's revenue segments according to its 10-K?", {"search_10k_report"}, "应路由到 TSLA.US 索引"),
    ("RAG", "微软的云业务在年报里怎么描述？", {"search_10k_report"}, "应路由到 MSFT.US 索引"),
    # === 混合调用类（5条）===
    ("混合", "苹果股价跌了，有什么风险值得关注？", {"get_stock_quote", "search_10k_report"}, "应同时包含价格和风险"),
    ("混合", "AAPL营收增长快吗？", {"get_financial_data"}, "应包含营收 + 增速"),
    ("混合", "腾讯现在的股价和净利润多少？", {"get_stock_quote", "get_financial_data"}, "应同时包含行情和财务"),
    ("混合", "Apple revenue and its main business risks", {"get_financial_data", "search_10k_report"}, "应包含营收数据和风险描述"),
    ("混合", "分析一下00700.HK：股价、营收", {"get_stock_quote", "get_financial_data"}, "应包含行情和财务数据"),
]


def score(test_case, output: str, tool_calls: list, observations: list) -> dict:
    _, _, expected_tools, _ = test_case
    called = {c["name"] for c in tool_calls}
    text = (output or "").lower()

    if called == expected_tools:
        tool_score = 2
    elif called & expected_tools:
        tool_score = 1
    else:
        tool_score = 0

    has_numbers = bool(re.search(r"\d", text))
    has_unit = bool(re.search(r"[$¥%]|美元|港元|亿|万|billion|million", text))
    accuracy_score = 2 if (has_numbers and has_unit) else (1 if has_numbers else 0)

    if "[来源" in output or "[source" in text or "longbridge" in text:
        source_score = 2
    elif "10-k" in text or "年报" in output or "来源" in output:
        source_score = 1
    else:
        source_score = 0

    errored = any(obs.startswith("[错误]") for obs in observations)
    relevance_score = 2 if (len(output) > 50 and not errored) else (1 if len(output) > 10 else 0)

    return {
        "工具调用": tool_score,
        "数值准确性": accuracy_score,
        "来源引用": source_score,
        "回答相关性": relevance_score,
        "总计": tool_score + accuracy_score + source_score + relevance_score,
    }


def main():
    limit = len(TEST_CASES)
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    cases = TEST_CASES[:limit]

    executor = build_agent(max_iterations=4)
    print(f"评测环境就绪，共 {len(cases)} 条用例\n{'=' * 60}")

    results = []
    for i, tc in enumerate(cases, 1):
        category, query, expected_tools, key_check = tc
        print(f"\n[{i}] {category} | {query}\n    预期工具: {sorted(expected_tools)}")
        executor.memory.clear()  # 每条独立评测
        t0 = time.time()
        try:
            r = executor.invoke({"input": query})
            output = r.get("output", "")
            steps = r.get("intermediate_steps", [])
            tool_calls = [{"name": action.tool, "input": action.tool_input} for action, _ in steps]
            observations = [str(obs) for _, obs in steps]
            s = score(tc, output, tool_calls, observations)
            note = ""
        except Exception as e:
            output, tool_calls, observations = "", [], []
            s = {d: 0 for d in EVAL_DIMS} | {"总计": 0}
            note = f"异常: {type(e).__name__}: {str(e)[:100]}"
            print(f"    {note}")

        print(f"    得分: {s['总计']}/8 | " + " ".join(f"{d}:{s[d]}" for d in EVAL_DIMS))
        print(f"    调用: {[c['name'] for c in tool_calls]}")
        print(f"    回答: {output[:150]}...")
        results.append({
            "序号": i, "类别": category, "查询": query, "预期工具": sorted(expected_tools),
            "关键验证点": key_check, "实际调用": tool_calls, **s,
            "回答": output, "耗时秒": round(time.time() - t0, 1), "备注": note,
        })

    print(f"\n{'=' * 60}\n评测结果汇总\n{'=' * 60}")
    print(f"用例数: {len(results)} | 平均得分: {sum(r['总计'] for r in results) / len(results):.1f}/8")
    summary = {}
    for dim in EVAL_DIMS:
        avg = sum(r[dim] for r in results) / len(results)
        full = sum(1 for r in results if r[dim] == 2)
        summary[dim] = {"平均分": round(avg, 2), "满分率": f"{full}/{len(results)}"}
        print(f"  {dim}: {avg:.2f}/2  满分 {full}/{len(results)}  {'█' * int(avg * 10)}{'░' * (20 - int(avg * 10))}")
    for cat in ["行情", "财报", "RAG", "混合"]:
        rs = [r for r in results if r["类别"] == cat]
        if rs:
            print(f"  {cat}: {sum(r['总计'] for r in rs) / len(rs):.1f}/8 ({len(rs)}条)")

    os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.EVAL_RESULTS_DIR, f"{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": config.LLM_MODEL, "embedding": config.EMBEDDING_MODEL,
                   "summary": summary, "cases": results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {path}")


if __name__ == "__main__":
    main()
