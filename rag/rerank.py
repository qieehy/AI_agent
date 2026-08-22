

from abc import ABC, abstractmethod

from .vector_store import SearchHit


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        ...
