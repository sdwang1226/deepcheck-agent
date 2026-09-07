"""端到端冒烟测试：数据源 → 索引 → 工具 → Agent，覆盖 3 只股票。用法：python test_agent.py"""
from deepcheck.agent import StepTracer, build_agent
from deepcheck.data import load_indices, load_longbridge, load_registry

TEST_QUERIES = [
    ("AAPL price", "AAPL current price?", "get_stock_quote"),
    ("NVDA 10-K risk", "What are NVIDIA's main risk factors from its 10-K annual report?", "search_10k_report"),
    ("TSLA 10-K revenue", "Search Tesla 10-K for revenue breakdown", "search_10k_report"),
]


def main():
    quote_ctx, fund_ctx = load_longbridge()
    print("1. Longbridge API OK")

    registry = load_registry()
    indices = load_indices(registry)
    print(f"2. FAISS RAG OK | {len(indices)} stocks: {', '.join(indices)}")

    executor = build_agent(quote_ctx, fund_ctx, indices, registry, callbacks=[StepTracer()])
    print("3. Agent OK")

    failed = []
    for label, query, expected_tool in TEST_QUERIES:
        print(f"\n4. [{label}] {query}")
        result = executor.invoke({"input": query})
        called = [action.tool for action, _ in result["intermediate_steps"]]
        ok = expected_tool in called and "[来源" in "".join(obs for _, obs in result["intermediate_steps"])
        print(f"   tools={called} -> {'PASS' if ok else 'FAIL'}")
        print(f"   {result['output'][:300]}")
        if not ok:
            failed.append(label)

    if failed:
        raise SystemExit(f"\nFAILED: {failed}")
    print("\nAll end-to-end checks PASSED")


if __name__ == "__main__":
    main()
