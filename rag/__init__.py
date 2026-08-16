from .embeddings import EmbeddingService, LocalEmbeddingService


def create_embedding_service(model_name: str | None = None) -> EmbeddingService:
    return LocalEmbeddingService(model_name or "BAAI/bge-small-zh-v1.5")

__all__ = ["create_embedding_service", "EmbeddingService", "LocalEmbeddingService"]
