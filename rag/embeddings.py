from abc import ABC, abstractmethod
from typing import cast


class EmbeddingService(ABC):
    """向量化服务接口"""
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """单条文本 → 向量"""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文本 → 向量列表（批量比逐条快 10x+）"""
        ...



class LocalEmbeddingService(EmbeddingService):
    """基于 sentence-transforming 的本地 BGE 模型"""
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("LocalEmbeddingService 需要 rag extras: pip install -e '.[rag]'") from e
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return cast(list[float], self._model.encode(text, normalize_embeddings=True).tolist())

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return cast(list[list[float]], self._model.encode(texts, normalize_embeddings=True).tolist())

