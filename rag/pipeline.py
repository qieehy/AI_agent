from dataclasses import dataclass, field

from llm import LLMClient

from .chunking import Chunk, chunk_text
from .embeddings import EmbeddingService
from .hybrid import BM25Index, rrf_fuse
from .loader import load_pdf
from .rerank import Reranker
from .vector_store import VectorStore


@dataclass(frozen=True)
class Source:
    id: str
    number: int
    score: float
    metadata: dict
    @property
    def text(self):
        return self.metadata["text"]
    @property
    def page(self):
        return self.metadata["page"]
    @property
    def source(self):
        return self.metadata["source"]

@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)

class RAGPipeline:
    """RAG 管线：索引 PDF 入向量库，检索 + 生成回答。

    依赖全部注入（embedder / store / LLM），不自行创建，便于测试替换。
    块 id 方案：{path}:{page}:{chunk_index}，可寻址、可按文档整体删除。
    _doc_chunks 记账：文档 -> 其块 id 列表；重复 index 同一文件幂等
    （先读旧账删除，成功后写新账）。
    """

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore, llm_client: LLMClient,
                 keyword_index: BM25Index | None = None, reranker: Reranker | None = None,
                 chunk_size=500, overlap=100):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.keyword_index = keyword_index
        self.reranker = reranker

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
                            **chunk.metadata,     #一开始chunk的局部 chunk_index
                            "source": path,
                            "page": page.page,
                            "text": chunk.text
                        }
                    )
                )


        texts = [chunk.text for chunk in chunks]

        vectors = self.embedding_service.embed_batch(texts)

        self.vector_store.delete(self._doc_chunks.get(path, []))     #若文件已处理过一次，删去原已经处理过的vectors

        if self.keyword_index is not None:
            self.keyword_index.delete(self._doc_chunks.get(path, []))

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

        if self.keyword_index is not None:
            for chunk in chunks:
                self.keyword_index.add(chunk.id, chunk.text, chunk.metadata)

        self._doc_chunks[path] = [chunk.id for chunk in chunks]

        return len(chunks)

    def ask(self, question: str, top_k: int = 3, fetch_k=None) -> Answer:
        """两段式检索 + 生成：召回 fetch 个候选，精排到 top_k，编号拼入 user 消息。

        - 召回段：dense 向量检索（+ 可选 BM25 关键词，RRF 融合），各取 fetch 个
          （fetch = fetch_k 或 top_k）。候选池要大于最终 top_k 才有精排空间
        - 精排段：注入 reranker 时对候选重新打分取 top_k；无 reranker 时
          fetch 退化为 top_k，一段式检索，行为与 D20 相同
        - [n] 是本次检索的临时编号，与块的全局 id 无关；编号让 LLM 回答携带
          引用标记（如 "[1]"），与 Answer.sources 的 number 一一对应
        - 无命中返回提示语 + 空 sources；LLM content 为空时 Answer.text 回退空串
        """
        fetch = fetch_k or top_k
        query_vector = self.embedding_service.embed(question)

        if self.keyword_index is not None:
            dense_hits = self.vector_store.search(query_vector, fetch)
            sparse_hits = self.keyword_index.search(question, fetch)
            hits = rrf_fuse(dense_hits, sparse_hits, fetch)
        else:
            hits = self.vector_store.search(query_vector, fetch)

        if not hits:
            return Answer(text="知识库中没有找到相关内容")

        if self.reranker is not None:
            hits = self.reranker.rerank(question, hits, top_k)

        sources = [Source(id=hit.id, number=i, score=hit.score, metadata=hit.metadata) for i, hit in enumerate(hits, start=1)]
        context = "\n\n".join([
            f"[{source.number}] {source.text}"
            for source in sources
        ])

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
        answer = Answer(text=response.choices[0].message.content or "", sources=sources)
        return answer


