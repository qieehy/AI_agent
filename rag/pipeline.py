from llm import LLMClient

from .chunking import Chunk, chunk_text
from .embeddings import EmbeddingService
from .loader import load_pdf
from .vector_store import VectorStore


class RAGPipeline:
    """RAG 管线：索引 PDF 入向量库，检索 + 生成回答。

    依赖全部注入（embedder / store / LLM），不自行创建，便于测试替换。
    块 id 方案：{path}:{page}:{chunk_index}，可寻址、可按文档整体删除。
    _doc_chunks 记账：文档 -> 其块 id 列表；重复 index 同一文件幂等
    （先读旧账删除，成功后写新账）。
    """

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore, llm_client: LLMClient, chunk_size=500, overlap=100):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_client = llm_client

        self.chunk_size = chunk_size
        self.overlap = overlap

        self._doc_chunks: dict[str, list[str]] = {}

    def index_pdf(self, path: str) -> int:
        """索引 PDF：加载 -> 逐页切块 -> 嵌入 -> 删除旧块 -> 写入新块。

        重复索引同一文件幂等：删除旧账中的块 id 后重新写入。
        记账契约：先读旧账（delete），全部成功后写新账（最后一行）。
        返回写入的块数。
        """
        pages = load_pdf(path)
        chunks = []

        for page in pages:

            page_chunks = chunk_text(
                page.text,
                self.chunk_size,
                self.overlap
            )
            # 重建 Chunk：chunk_text 只给块内 id 和 chunk_index；
            # 此处补全局 id（{path}:{page}:{idx}）及 source/page/text 元数据
            # （text 是检索命中后生成答案的上下文来源）。
            for chunk in page_chunks:
                chunks.append(
                    Chunk(
                        id=f"{path}:{page.page}:{chunk.metadata['chunk_index']}",
                        text=chunk.text,
                        metadata={
                            **chunk.metadata,
                            "source": path,
                            "page": page.page,
                            "text": chunk.text
                        }
                    )
                )


        texts = [chunk.text for chunk in chunks]

        vectors = self.embedding_service.embed_batch(texts)

        self.vector_store.delete(self._doc_chunks.get(path, []))

        self.vector_store.add(
            ids=[
                chunk.id
                for chunk in chunks
            ],

            vectors=vectors,

            metadata=[
                chunk.metadata
                for chunk in chunks
            ]
        )
        self._doc_chunks[path] = [chunk.id for chunk in chunks]

        return len(chunks)

    def ask(self, question: str, top_k: int = 3) -> str:
        """检索 top_k 个块，按 [1..n] 编号拼入 user 消息，由 LLM 依据上下文回答。

        编号让 LLM 回答携带引用标记（如 "[1]"），是 D21 citation 的种子；
        [n] 是本次检索的临时编号，与块的全局 id 无关。
        无命中返回提示语；返回 LLM 文本（content 为空时返回空串）。
        """
        query_vector = self.embedding_service.embed(question)

        hits = self.vector_store.search(query_vector, top_k)
        if not hits:
            return "知识库中没有找到相关内容"

        contexts = [
            f"[{i}] {hit.metadata['text']}"
            for i, hit in enumerate(hits, start=1)
        ]

        context = "\n\n".join(contexts)

        messages = [
            {
                "role": "system",
                "content":
                    (
                        "你是知识库助手。"
                        "只能根据上下文回答问题。"
                        "不知道就说不知道, 不要自己猜想。"
                        "回答时请用 [编号] 标注你引用了哪段上下文。"
                    )
            },
            {
                "role": "user",
                "content":
                    f"""
        上下文:
        {context}
        问题:
        {question}
        """
            }
        ]
        response = self.llm_client.chat(messages)
        content = response.choices[0].message.content
        return content or ""


