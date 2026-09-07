from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

from deepcheck import config


class BGEEmbeddings(Embeddings):
    """BAAI/bge-large-zh-v1.5：1024 维、中英双语，本地 CPU 推理。"""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model = SentenceTransformer(
            model_name, device="cpu", local_files_only=config.EMBEDDING_LOCAL_ONLY
        )

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()
