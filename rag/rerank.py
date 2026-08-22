from abc import ABC, abstractmethod

from .vector_store import SearchHit


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        ...

class CrossEncoderReranker(Reranker):
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", max_length=1024):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name=model_name, max_length=max_length)
        except ImportError as e:
            raise ImportError("reranker模型导入失败: 请先安装Cross Encoder模型") from e

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []

        pairs = [(query, hit.text) for hit in hits]
        scores = self._model.predict(pairs)
        ranked = [
            SearchHit(
                id=hit.id,
                score=float(score),
                metadata=hit.metadata
            )
            for score, hit in zip(scores, hits, strict=True)
        ]
        ranked.sort(key=lambda hit: hit.score, reverse=True)

        return ranked[:top_k]
