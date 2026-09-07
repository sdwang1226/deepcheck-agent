"""Agent 装配：Tool Calling Agent + 多轮记忆 + 执行过程回调。"""
import os
import time

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks import BaseCallbackHandler
from langchain_deepseek import ChatDeepSeek

from deepcheck import config
from deepcheck.data import load_indices, load_longbridge, load_registry
from deepcheck.tools import build_tools

SYSTEM_PROMPT = """You are DeepCheck, a financial research assistant.

RULES:
1. If the user does NOT specify a stock, use the SAME stock from the conversation history.
   If there is no stock in the history either, ASK the user which stock they mean instead of guessing.
2. Every answer must cite its data source, e.g. "[来源: Longbridge Quote API]". Never drop the source prefix returned by tools.
3. NEVER make up numbers. If a tool returns no data or an error, say so explicitly.
4. Format large numbers for readability: "$111.18B" instead of "111184000000.0000".
5. 10-K reports are available for: {available_stocks}. Always pass both `query` and `symbol` when searching them.
   Translate Chinese queries into English before searching, because 10-K filings are in English.
6. Answer in the same language the user used."""


class StepTracer(BaseCallbackHandler):
    """把 Agent 的每一步（工具、参数、耗时、来源）以结构化形式打印出来，模拟客户端的"执行反馈"区。"""

    def __init__(self, stream=print):
        self.stream = stream
        self.steps = []
        self._start = None

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._start = time.time()
        name = serialized.get("name", "tool")
        self.stream(f"  → [步骤 {len(self.steps) + 1}] 调用 {name}({input_str})")

    def on_tool_end(self, output, **kwargs):
        elapsed = time.time() - self._start if self._start else 0.0
        text = output if isinstance(output, str) else str(output)
        source = text.split("\n", 1)[0] if text.startswith("[") else "(无来源标注)"
        self.steps.append({"source": source, "elapsed": round(elapsed, 2)})
        self.stream(f"  ← 完成，耗时 {elapsed:.1f}s，{source}")


def build_agent(quote_ctx=None, fund_ctx=None, stock_indices=None, stock_registry=None,
                verbose=False, max_iterations=8, callbacks=None):
    if quote_ctx is None or fund_ctx is None:
        quote_ctx, fund_ctx = load_longbridge()
    if stock_registry is None:
        stock_registry = load_registry()
    if stock_indices is None:
        stock_indices = load_indices(stock_registry)

    tools = build_tools(quote_ctx, fund_ctx, stock_indices, stock_registry)
    available = ", ".join(f"{t} ({info['name']})" for t, info in stock_registry.items())

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT.format(available_stocks=available)),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    llm = ChatDeepSeek(model=config.LLM_MODEL, api_key=os.getenv("DEEPSEEK_API_KEY"), temperature=0)
    agent = create_tool_calling_agent(llm, tools, prompt)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=max_iterations,
        return_intermediate_steps=True,
        callbacks=callbacks or [],
    )
