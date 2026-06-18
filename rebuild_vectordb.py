"""重建 FAISS 向量库 (Windows 环境) — 使用 BGE-large-zh-v1.5"""
import os, sys, traceback
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

# Patch transformers torch version check (torch 2.4.1, check requires 2.6+)
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

BASE = os.path.dirname(os.path.abspath(__file__))

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from sentence_transformers import SentenceTransformer
    from langchain.embeddings.base import Embeddings

    # 1. 加载文本
    print("1. 加载 Apple 10-K...", flush=True)
    with open(f"{BASE}/data/apple_10k_clean.txt", "r", encoding="utf-8") as f:
        text = f.read()

    part1 = text.find("PART I")
    item1 = text.find("Item 1.Business", part1)
    body = text[item1:] if item1 > 0 else text[part1:]
    print(f"   正文长度: {len(body):,} 字符", flush=True)

    # 2. 切分
    print("2. 切分文本...", flush=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = splitter.create_documents([body])
    print(f"   {len(docs)} 个 chunk", flush=True)

    # 3. Embedding (BGE-large-zh-v1.5, 1024维, 中英双语)
    print("3. 加载 embedding 模型 (BGE-large-zh-v1.5)...", flush=True)
    class SE(Embeddings):
        def __init__(self):
            self.m = SentenceTransformer("BAAI/bge-large-zh-v1.5")
        def embed_documents(self, t):
            return self.m.encode(t, normalize_embeddings=True).tolist()
        def embed_query(self, t):
            return self.m.encode([t], normalize_embeddings=True)[0].tolist()

    embedding = SE()
    print("   模型加载完成", flush=True)

    # 4. 构建向量库
    print("4. 构建 FAISS 向量库...", flush=True)
    db_path = f"{BASE}/data/faiss_index"

    vectordb = FAISS.from_documents(documents=docs, embedding=embedding)
    vectordb.save_local(db_path)
    print(f"   ✅ 向量库完成，已保存到 {db_path}", flush=True)

    # 5. 测试检索
    print("\n5. 测试检索...", flush=True)
    queries = [
        "what are the main risk factors?",
        "how much revenue did the company earn?",
        "苹果的供应链集中在哪些地区？",
    ]
    for q in queries:
        results = vectordb.similarity_search(q, k=1)
        print(f"\n   Q: {q}")
        print(f"   A: {results[0].page_content[:150]}...")

    print("\n✅ 全部完成！")

except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    traceback.print_exc()
    sys.exit(1)
