"""对比 MiniLM vs BGE-large-zh: 高难度测试（中文查询 + 语义匹配）"""
import os, sys, time, json
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

BASE = os.path.dirname(os.path.abspath(__file__))

# 高难度测试：中文查询 + 语义匹配 + 需要理解意图的查询
hard_tests = [
    # --- 中文查询（模型需理解跨语言语义）---
    ("苹果公司面临的地缘政治风险有哪些？",
     ["geopolitical", "international", "trade", "tariff", "China", "country", "government"],
     "地缘政治风险"),
    ("苹果的供应链集中在哪些地区？有什么隐患？",
     ["supply chain", "supplier", "concentration", "China", "manufacturing", "single source"],
     "供应链集中风险"),
    ("苹果在研发方面投入了多少钱？",
     ["research", "development", "R&D", "expenditure", "billion"],
     "研发投入"),
    ("苹果的现金流状况如何？",
     ["cash", "operating", "free cash flow", "liquidity", "capital"],
     "现金流"),
    ("苹果面临哪些知识产权方面的风险？",
     ["intellectual property", "patent", "copyright", "trademark", "infringement"],
     "知识产权"),

    # --- 需要语义理解的英文查询（不是简单关键词）---
    ("What could cause Apple's stock price to drop significantly?",
     ["risk", "decline", "adverse", "material", "volatility", "stock price", "market"],
     "股价下跌因素"),
    ("How dependent is Apple on iPhone sales?",
     ["iPhone", "product", "concentration", "significant portion", "revenue"],
     "iPhone依赖度"),
    ("What environmental commitments has Apple made?",
     ["environment", "carbon", "climate", "renewable", "sustainability", "emission"],
     "环保承诺"),
    ("Who are Apple's main competitors?",
     ["compet", "Samsung", "Google", "Microsoft", "Android", "market"],
     "主要竞争对手"),
    ("What is Apple's dividend and share buyback policy?",
     ["dividend", "repurchase", "buyback", "return", "shareholder", "capital return"],
     "分红回购政策"),
]

def load_index(model_name, db_path):
    class Emb(Embeddings):
        def __init__(self):
            self.m = SentenceTransformer(model_name)
        def embed_documents(self, t):
            return self.m.encode(t, normalize_embeddings=True).tolist()
        def embed_query(self, t):
            return self.m.encode([t], normalize_embeddings=True)[0].tolist()
    emb = Emb()
    vectordb = FAISS.load_local(db_path, emb, allow_dangerous_deserialization=True)
    return vectordb, emb

def eval_hard(name, vectordb):
    print(f"\n{'='*60}", flush=True)
    print(f"Model: {name}", flush=True)
    print(f"{'='*60}", flush=True)
    hits = 0
    results_detail = []
    for q, keywords, desc in hard_tests:
        results = vectordb.similarity_search(q, k=3)
        combined = " ".join([r.page_content for r in results]).lower()
        hit = any(kw.lower() in combined for kw in keywords)
        hits += hit
        status = "✅" if hit else "❌"
        print(f"   {status} [{desc}] {q[:50]}", flush=True)
        if not hit:
            print(f"      Expected any of: {keywords[:4]}...", flush=True)
            print(f"      Top result: {results[0].page_content[:120]}...", flush=True)
        results_detail.append({"query": q, "desc": desc, "hit": hit})
    rate = hits / len(hard_tests) * 100
    print(f"\n   Hard hit rate: {hits}/{len(hard_tests)} = {rate:.0f}%", flush=True)
    return rate, results_detail

# ===== Run =====
print("Loading MiniLM index...", flush=True)
vdb1, _ = load_index("sentence-transformers/all-MiniLM-L6-v2", f"{BASE}/data/faiss_index_minilm")
r1, d1 = eval_hard("MiniLM (384d)", vdb1)

print("\nLoading BGE-large index...", flush=True)
vdb2, _ = load_index("BAAI/bge-large-zh-v1.5", f"{BASE}/data/faiss_index_bge")
r2, d2 = eval_hard("BGE-large-zh (1024d)", vdb2)

# ===== Summary =====
print(f"\n{'='*60}", flush=True)
print(f"HARD TEST COMPARISON", flush=True)
print(f"{'='*60}", flush=True)
print(f"  MiniLM:    {r1:.0f}% ({sum(1 for x in d1 if x['hit'])}/{len(d1)})", flush=True)
print(f"  BGE-large: {r2:.0f}% ({sum(1 for x in d2 if x['hit'])}/{len(d2)})", flush=True)
print(f"  Delta:     {r2-r1:+.0f}%", flush=True)

# Per-query comparison
print(f"\n  Per-query breakdown:", flush=True)
print(f"  {'Query':<35} {'MiniLM':<10} {'BGE':<10}", flush=True)
print(f"  {'-'*55}", flush=True)
for i in range(len(hard_tests)):
    desc = hard_tests[i][2]
    m_hit = "✅" if d1[i]["hit"] else "❌"
    b_hit = "✅" if d2[i]["hit"] else "❌"
    print(f"  {desc:<35} {m_hit:<10} {b_hit:<10}", flush=True)

print(f"\n✅ 对比完成！", flush=True)
