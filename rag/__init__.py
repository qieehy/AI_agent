from .embeddings import EmbeddingService, LocalEmbeddingService
from .vector_store import FAISSVectorStore, SearchHit, VectorStore


def create_embedding_service(model_name: str | None = None) -> EmbeddingService:
    return LocalEmbeddingService(model_name or "BAAI/bge-small-zh-v1.5")

def create_vector_store() -> VectorStore:
    return FAISSVectorStore()

__all__ = ["create_embedding_service", "EmbeddingService", "LocalEmbeddingService", "create_vector_store", "FAISSVectorStore", "VectorStore", "SearchHit"]
