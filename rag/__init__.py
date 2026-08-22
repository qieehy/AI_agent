from .embeddings import EmbeddingService, LocalEmbeddingService
from .hybrid import BM25Index
from .pipeline import RAGPipeline
from .rerank import CrossEncoderReranker, Reranker
from .vector_store import FAISSVectorStore, SearchHit, VectorStore


def create_embedding_service(model_name: str | None = None) -> EmbeddingService:
    return LocalEmbeddingService(model_name or "BAAI/bge-small-zh-v1.5")

def create_vector_store() -> VectorStore:
    return FAISSVectorStore()

def create_reranker() -> Reranker:
    return CrossEncoderReranker()

def create_rag_pipeline(embedding_service, vector_store, llm_client,
                        keyword_index=None, reranker=None, chunk_size=500, overlap=100) -> RAGPipeline:
    return RAGPipeline(embedding_service=embedding_service, vector_store=vector_store, llm_client=llm_client,
                       keyword_index=keyword_index, reranker=reranker,
                       chunk_size=chunk_size, overlap=overlap)

__all__ = ["create_embedding_service", "EmbeddingService", "LocalEmbeddingService", "create_vector_store", "FAISSVectorStore", "VectorStore", "SearchHit",
           "RAGPipeline", "create_rag_pipeline", "BM25Index", "create_reranker", "Reranker", "CrossEncoderReranker"]
