"""从 SEC EDGAR 下载多只股票的 10-K 年报并构建 FAISS 索引"""
import os, sys, json, time, re, traceback
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = f"{BASE}/data/10k_filings"
INDEX_DIR = f"{BASE}/data/faiss_indices"

# SEC EDGAR requires User-Agent header
HEADERS = {
    "User-Agent": "DeepCheck Research Agent research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# 目标股票: ticker -> (company_name, CIK)
# CIK from SEC EDGAR company search
STOCKS = {
    "AAPL.US": ("Apple Inc.", "0000320193"),
    "MSFT.US": ("Microsoft Corporation", "0000789019"),
    "GOOGL.US": ("Alphabet Inc.", "0001652044"),
    "NVDA.US": ("NVIDIA Corporation", "0001045810"),
    "TSLA.US": ("Tesla Inc.", "0001318605"),
    "AMZN.US": ("Amazon.com Inc.", "0001018724"),
}

def download_10k(ticker, company_name, cik):
    """从 SEC EDGAR 下载最新 10-K 全文"""
    filepath = f"{DATA_DIR}/{ticker.replace('.', '_')}_10k.txt"
    if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
        print(f"   [缓存] {filepath} 已存在，跳过下载", flush=True)
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    print(f"   下载 {company_name} ({cik}) 10-K...", flush=True)

    # Step 1: 获取最新 10-K filing accession number
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    time.sleep(0.2)  # SEC rate limit: 10 req/sec
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # 在 recent filings 中找 10-K
    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])

    accession = None
    primary_doc = None
    for i, form in enumerate(forms):
        if form == "10-K":
            accession = accessions[i]
            primary_doc = primary_docs[i]
            break

    if not accession:
        print(f"   ❌ 未找到 10-K filing for {ticker}", flush=True)
        return None

    # Step 2: 下载 10-K 全文 (HTML)
    acc_no_dash = accession.replace("-", "")
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc_no_dash}/{primary_doc}"
    print(f"   URL: {doc_url}", flush=True)

    time.sleep(0.2)
    resp = requests.get(doc_url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    html = resp.text

    # Step 3: 清洗 HTML -> 纯文本
    text = clean_html(html)
    print(f"   原文长度: {len(text):,} 字符", flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"   已保存: {filepath}", flush=True)
    return text

def clean_html(html):
    """简单 HTML -> 纯文本清洗"""
    # 去掉 style/script 标签及内容
    text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # 表格行变换行
    text = re.sub(r'</tr>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</td>', ' | ', text, flags=re.IGNORECASE)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</div>', '\n', text, flags=re.IGNORECASE)
    # 去掉所有 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去掉 HTML 实体
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'&\w+;', ' ', text)
    # 压缩空行
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_body(text):
    """提取 10-K 正文（跳过 SEC 头部）"""
    # 尝试找 PART I 或 Item 1
    part1 = text.find("PART I")
    if part1 == -1:
        part1 = text.find("Part I")
    if part1 == -1:
        part1 = text.find("ITEM 1")
    if part1 == -1:
        # 跳过前 5% 的 header
        part1 = len(text) // 20
    return text[part1:]

def build_index(ticker, company_name, text, embedding):
    """为单只股票构建 FAISS 索引"""
    body = extract_body(text)
    print(f"   正文长度: {len(body):,} 字符", flush=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    # 每个 chunk 带 metadata 标注来源
    docs = splitter.create_documents(
        [body],
        metadatas=[{"source": f"{company_name} 10-K", "ticker": ticker}]
    )
    print(f"   {len(docs)} 个 chunk", flush=True)

    index_path = f"{INDEX_DIR}/{ticker.replace('.', '_')}"
    os.makedirs(index_path, exist_ok=True)

    vectordb = FAISS.from_documents(documents=docs, embedding=embedding)
    vectordb.save_local(index_path)
    print(f"   ✅ 索引已保存: {index_path}", flush=True)
    return len(docs)

# ===== Main =====
if __name__ == "__main__":
    print("="*60, flush=True)
    print("多股票 10-K 年报入库", flush=True)
    print("="*60, flush=True)

    # 初始化 embedding
    print("\n1. 加载 BGE-large-zh-v1.5 embedding...", flush=True)
    class SE(Embeddings):
        def __init__(self):
            self.m = SentenceTransformer("BAAI/bge-large-zh-v1.5")
        def embed_documents(self, t):
            return self.m.encode(t, normalize_embeddings=True).tolist()
        def embed_query(self, t):
            return self.m.encode([t], normalize_embeddings=True)[0].tolist()

    embedding = SE()
    print("   模型加载完成", flush=True)

    # 处理 Apple（使用已有的清洗文本）
    print(f"\n2. 处理各股票 10-K...", flush=True)
    registry = {}
    total_chunks = 0

    # Apple 使用已有文件
    print(f"\n--- AAPL.US (Apple Inc.) ---", flush=True)
    aapl_path = f"{BASE}/data/apple_10k_clean.txt"
    with open(aapl_path, "r", encoding="utf-8") as f:
        aapl_text = f.read()
    n = build_index("AAPL.US", "Apple Inc.", aapl_text, embedding)
    registry["AAPL.US"] = {"name": "Apple Inc.", "chunks": n, "index_path": f"{INDEX_DIR}/AAPL_US"}
    total_chunks += n

    # 其他股票从 SEC EDGAR 下载
    for ticker, (name, cik) in STOCKS.items():
        if ticker == "AAPL.US":
            continue
        print(f"\n--- {ticker} ({name}) ---", flush=True)
        try:
            text = download_10k(ticker, name, cik)
            if text and len(text) > 5000:
                n = build_index(ticker, name, text, embedding)
                registry[ticker] = {
                    "name": name,
                    "chunks": n,
                    "index_path": f"{INDEX_DIR}/{ticker.replace('.', '_')}"
                }
                total_chunks += n
            else:
                print(f"   ⚠️ 文本过短或下载失败，跳过", flush=True)
        except Exception as e:
            print(f"   ❌ ERROR: {e}", flush=True)
            traceback.print_exc()

    # 保存注册表
    registry_path = f"{INDEX_DIR}/registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"入库完成！", flush=True)
    print(f"  股票数: {len(registry)}", flush=True)
    print(f"  总 chunks: {total_chunks}", flush=True)
    print(f"  注册表: {registry_path}", flush=True)
    for t, info in registry.items():
        print(f"  {t}: {info['name']} ({info['chunks']} chunks)", flush=True)
    print(f"\n✅ 全部完成！", flush=True)
