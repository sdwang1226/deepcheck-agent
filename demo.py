"""
DeepCheck Agent — 交互式演示
============================
用法：
  python demo.py            # 交互对话；输入 demo 跑预设场景，输入 memory 跑多轮记忆场景，q 退出
  python demo.py --demo     # 直接跑预设场景后退出
  python demo.py --memory   # 直接跑多轮记忆场景后退出
"""
import sys

from deepcheck.agent import StepTracer, build_agent
from deepcheck.data import load_indices, load_longbridge, load_registry

DEMO_QUERIES = [
    ("[1] 实时行情", "AAPL current price?"),
    ("[2] NVIDIA 年报风险", "What are NVIDIA's main risk factors from its 10-K annual report?"),
    ("[3] 多轮记忆", "Its latest revenue and EPS?"),
    ("[4] Tesla 年报检索", "Search Tesla 10-K for revenue breakdown by segment"),
]

# 观察要点：Round 2/4/6 没指定股票，Agent 应沿用上一轮的股票；Round 0 历史为空时应反问而非猜测
MEMORY_QUERIES = [
    ("Round 0: 无历史、无股票 → 应反问", "Its main risk factors?"),
    ("Round 1: 指定 AAPL", "AAPL current price and latest revenue?"),
    ("Round 2: 不指定股票 → 沿用 AAPL", "Its main risk factors from 10-K?"),
    ("Round 3: 切换到 NVDA", "What are NVIDIA's main risk factors from its 10-K?"),
    ("Round 4: 不指定股票 → 沿用 NVDA", "Its revenue breakdown?"),
    ("Round 5: 切换到腾讯", "Now check 00700.HK Tencent price"),
    ("Round 6: 不指定股票 → 沿用 00700.HK", "Its financial data?"),
]


def run_scenario(executor, queries):
    for label, query in queries:
        print(f"\n{'=' * 60}\n  {label}\n  Query: {query}\n{'=' * 60}")
        result = executor.invoke({"input": query})
        print(f"\nAgent:\n{result['output']}")
    print(f"\n{'=' * 60}\n  [OK] 场景完成\n{'=' * 60}")


def main():
    print("=" * 60)
    print("  DeepCheck — AI 投研尽调 Agent")
    print("  Tech: DeepSeek Tool Calling + Longbridge OpenAPI + FAISS RAG")
    print("=" * 60)
    print("\n[*] 加载模型与索引...")

    quote_ctx, fund_ctx = load_longbridge()
    registry = load_registry()
    indices = load_indices(registry)
    print(f"[OK] 就绪 | {len(indices)} 只股票年报: {', '.join(registry)}\n")

    executor = build_agent(quote_ctx, fund_ctx, indices, registry, callbacks=[StepTracer()])

    if "--demo" in sys.argv:
        return run_scenario(executor, DEMO_QUERIES)
    if "--memory" in sys.argv:
        return run_scenario(executor, MEMORY_QUERIES)

    print("-" * 60)
    print(">> 输入问题开始对话（demo: 预设场景 | memory: 多轮记忆场景 | q: 退出）")
    print("-" * 60)
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() == "q":
            break
        if user_input.lower() == "demo":
            run_scenario(executor, DEMO_QUERIES)
            continue
        if user_input.lower() == "memory":
            run_scenario(executor, MEMORY_QUERIES)
            continue
        result = executor.invoke({"input": user_input})
        print(f"\nAgent:\n{result['output']}")
    print("\nBye!")


if __name__ == "__main__":
    main()
