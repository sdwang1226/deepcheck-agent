"""对比 MiniLM vs BGE-large-zh embedding 模型的 RAG 命中率"""
import os, sys, time
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

# Patch transformers torch version check (we use torch 2.4.1, check requires 2.6+)
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

BASE = os.path.dirname(os.path.abspath(__file__))

# ===== 1. 加载文本 + 切分 =====
print("1. 加载 Apple 10-K + 切分...", flush=True)
with open(f"{BASE}/data/apple_10k_clean.txt", "r", encoding="utf-8") as f:
    text = f.read()
part1 = text.find("PART I")
item1 = text.find("Item 1.Business", part1)
body = text[item1:] if item1 > 0 else text[part1:]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)
docs = splitter.create_documents([body])
print(f"   {len(docs)} chunks, {len(body):,} chars", flush=True)

# ===== 2. 定义测试用例 =====
# 每条 = (query, expected_keywords_in_result)
test_cases = [
    ("What are Apple's main risk factors?",
     ["risk", "adverse", "materialize"]),
    ("How much revenue did Apple earn?",
     ["revenue", "net sales", "million", "billion"]),
    ("How many employees does Apple have?",
     ["employee", "human capital", "approximately"]),
    ("What is Apple's strategy for artificial intelligence?",
     ["machine learning", "artificial intelligence", "AI", "technology"]),
    ("What legal proceedings is Apple involved in?",
     ["legal", "litigation", "proceeding", "court", "Epic"]),
]

def eval_model(model_name, db_save_path):
    """Build FAISS index and evaluate RAG hit rate."""
    print(f"\n{'='*60}", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"{'='*60}", flush=True)

    class Emb(Embeddings):
        def __init__(self):
            self.m = SentenceTransformer(model_name)
        def embed_documents(self, t):
            return self.m.encode(t, normalize_embeddings=True).tolist()
        def embed_query(self, t):
            return self.m.encode([t], normalize_embeddings=True)[0].tolist()

    emb = Emb()
    dim = emb.m.get_sentence_embedding_dimension()
    print(f"   Dimension: {dim}", flush=True)

    t0 = time.time()
    vectordb = FAISS.from_documents(documents=docs, embedding=emb)
    vectordb.save_local(db_save_path)
    build_time = time.time() - t0
    print(f"   Index built in {build_time:.1f}s → {db_save_path}", flush=True)

    hits = 0
    print(f"\n   Testing {len(test_cases)} queries:", flush=True)
    for q, keywords in test_cases:
        results = vectordb.similarity_search(q, k=2)
        combined = " ".join([r.page_content for r in results]).lower()
        hit = any(kw.lower() in combined for kw in keywords)
        hits += hit
        status = "✅ HIT" if hit else "❌ MISS"
        print(f"   {status} | Q: {q}", flush=True)
        if not hit:
            print(f"         Expected keywords: {keywords}", flush=True)
            print(f"         Got: {combined[:200]}...", flush=True)

    rate = hits / len(test_cases) * 100
    print(f"\n   Hit rate: {hits}/{len(test_cases)} = {rate:.0f}%", flush=True)
    return rate, build_time, dim

# ===== 3. 对比两个模型 =====
print("\n" + "="*60, flush=True)
print("Starting comparison...", flush=True)

r1, t1, d1 = eval_model(
    "sentence-transformers/all-MiniLM-L6-v2",
    f"{BASE}/data/faiss_index_minilm"
)

r2, t2, d2 = eval_model(
    "BAAI/bge-large-zh-v1.5",
    f"{BASE}/data/faiss_index_bge"
)

# ===== 4. 结果汇总 =====
print(f"\n{'='*60}", flush=True)
print(f"COMPARISON RESULTS", flush=True)
print(f"{'='*60}", flush=True)
print(f"{'Metric':<25} {'MiniLM (v1)':<20} {'BGE-large (v2)':<20}", flush=True)
print(f"{'-'*65}", flush=True)
print(f"{'Dimension':<25} {d1:<20} {d2:<20}", flush=True)
print(f"{'Build time':<25} {t1:<20.1f} {t2:<20.1f}", flush=True)
print(f"{'Hit rate':<25} {r1:<20.0f}% {r2:<20.0f}%", flush=True)
improvement = r2 - r1
print(f"{'Improvement':<25} {'—':<20} {'+' if improvement >= 0 else ''}{improvement:.0f}%", flush=True)
print(f"\n✅ 对比完成！", flush=True)
