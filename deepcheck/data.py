"""数据源接入：Longbridge OpenAPI 上下文 + 多股票 FAISS 索引加载。"""
import json
import os

from langchain_community.vectorstores import FAISS
from longbridge.openapi import Config, FundamentalContext, QuoteContext

from deepcheck import config


def load_longbridge():
    cfg = Config.from_apikey(
        app_key=os.getenv("LONGBRIDGE_APP_KEY"),
        app_secret=os.getenv("LONGBRIDGE_APP_SECRET"),
        access_token=os.getenv("LONGBRIDGE_ACCESS_TOKEN"),
    )
    return QuoteContext(cfg), FundamentalContext(cfg)


def load_registry(path: str = config.REGISTRY_PATH) -> dict:
    """registry.json: {"AAPL.US": {"name": "Apple Inc.", "chunks": 302, "index_path": "..."}}"""
    with open(path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    for info in registry.values():
        if not os.path.isabs(info["index_path"]):
            info["index_path"] = os.path.join(config.BASE_DIR, info["index_path"])
    return registry


def load_indices(registry: dict, embedding=None) -> dict:
    if embedding is None:
        from deepcheck.embeddings import (
            BGEEmbeddings,  # 延迟导入：加载 torch 较重，仅在真正需要索引时触发
        )

        embedding = BGEEmbeddings()
    return {
        ticker: FAISS.load_local(info["index_path"], embedding, allow_dangerous_deserialization=True)
        for ticker, info in registry.items()
    }
