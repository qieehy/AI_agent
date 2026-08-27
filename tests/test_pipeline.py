"""D20: RAG pipeline 测试。

策略（延续 D18/D19 分层）：
- 语义质量在 D18（真 BGE）、检索机制在 D19（合成向量）已测；
  这里只测"管线把三件事接对了"——编排层
- FakeEmbedder：确定性向量（由文本长度派生），不加载 BGE
- FakeVectorStore：内存假向量库，不碰 faiss——DI 让编排测试摆脱重依赖，
  CI 直接运行；真 FAISS 接线由 test_vector_store（本地）+ demo 手工验收覆盖
- mock LLM：只捕获 prompt、返回预设答案，不调 API
- load_pdf 用 monkeypatch 换掉，不碰真实文件
"""
from types import SimpleNamespace

import pytest

import rag.pipeline as pipeline_module
from rag import SearchHit, create_rag_pipeline
from rag.hybrid import BM25Index
from rag.loader import PageText


class FakeEmbedder:
    """确定性假嵌入：向量由文本长度派生（4 维），不加载 BGE。"""

    def embed(self, text: str) -> list[float]:
        return [1.0, float(len(text) % 5), 0.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class FakeVectorStore:
    """内存假向量库：实现 VectorStore 接口，供编排测试用，不碰 faiss。"""

    def __init__(self):
        self._entries: dict[str, tuple[list[float], dict]] = {}

    def add(self, ids, vectors, metadata=None):
        metadata = metadata or [{} for _ in ids]
        for id_, vec, meta in zip(ids, vectors, metadata, strict=True):
            self._entries[id_] = (vec, meta)

    def search(self, query, top_k=5):
        return [
            SearchHit(id=k, score=1.0, metadata=m)
            for k, (_, m) in self._entries.items()
        ][:top_k]

    def delete(self, ids):
        for id_ in ids:
            self._entries.pop(id_, None)

    def update(self, ids, vectors, metadata=None):
        self.delete(ids)
        self.add(ids, vectors, metadata)


@pytest.fixture
def env(mocker):
    """FakeVectorStore + fake embedder + async mock LLM（D24: ask 已是 async）。"""
    store = FakeVectorStore()
    llm = mocker.AsyncMock()
    llm.chat.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="标准答案[1]"))]
    )
    pipeline = create_rag_pipeline(FakeEmbedder(), store, llm)
    return SimpleNamespace(pipeline=pipeline, store=store, llm=llm)


@pytest.fixture
def hybrid_env(env):
    """env + 真 BM25Index 接入 pipeline（BM25 纯 Python 轻量，用真的不 fake）。

    经 create_rag_pipeline(..., keyword_index=...) 构造，
    同时钉死工厂签名的 keyword_index 透传。
    """
    keyword_index = BM25Index()
    pipeline = create_rag_pipeline(
        FakeEmbedder(), env.store, env.llm, keyword_index=keyword_index
    )
    return SimpleNamespace(
        pipeline=pipeline, store=env.store, llm=env.llm, keyword_index=keyword_index
    )


class FakeReranker:
    """按块文本长度降序的确定性假 reranker——编排测试不碰 2.3GB 模型，CI 直接跑。"""

    def rerank(self, query, hits, top_k):
        return sorted(hits, key=lambda h: len(h.metadata["text"]), reverse=True)[:top_k]


@pytest.fixture
def rerank_env(env):
    """env + FakeReranker 接入（两段式编排，零重依赖）。"""
    reranker = FakeReranker()
    pipeline = create_rag_pipeline(
        FakeEmbedder(), env.store, env.llm, reranker=reranker
    )
    return SimpleNamespace(
        pipeline=pipeline, store=env.store, llm=env.llm, reranker=reranker
    )


def _patch_pdf(monkeypatch, pages: list[str]):
    monkeypatch.setattr(
        pipeline_module,
        "load_pdf",
        lambda path: [PageText(page=i + 1, text=t) for i, t in enumerate(pages)],
    )


# ---------- index ----------

def test_index_pdf_returns_chunk_count(env, monkeypatch):
    """索引返回块数；8 块 = 3000 字 / 步长 400。"""
    _patch_pdf(monkeypatch, ["x" * 3000])

    assert env.pipeline.index_pdf("doc.pdf") == 8


def test_reindex_replaces_old_chunks(env, monkeypatch):
    """重复索引幂等：v1 长文档 8 块 -> v2 短文档 3 块，库里只剩新块。"""
    _patch_pdf(monkeypatch, ["x" * 3000])
    env.pipeline.index_pdf("doc.pdf")

    _patch_pdf(monkeypatch, ["y" * 1000])
    assert env.pipeline.index_pdf("doc.pdf") == 3

    hits = env.store.search([1.0, 0.0, 0.0, 0.0], top_k=20)
    assert len(hits) == 3


def test_chunk_metadata_carries_source_page_text(env, monkeypatch):
    """检索命中的 metadata 携带 source/page/text（生成答案的上下文来源）。"""
    _patch_pdf(monkeypatch, ["第一页内容", "第二页内容"])
    env.pipeline.index_pdf("report.pdf")

    hits = env.store.search([1.0, 0.0, 0.0, 0.0], top_k=20)

    for hit in hits:
        assert hit.metadata["source"] == "report.pdf"
        assert hit.metadata["page"] in (1, 2)
        assert hit.metadata["text"] in ("第一页内容", "第二页内容")
    # 分数相等时 FAISS 返回顺序无定义，只断言集合
    assert {h.id for h in hits} == {"report.pdf:1:0", "report.pdf:2:0"}


# ---------- ask（D24: ask 已是 async，全部 await） ----------

@pytest.mark.anyio
async def test_ask_returns_llm_text(env, monkeypatch):
    """LLM 的 content 原样进入 Answer.text，sources 非空。"""
    _patch_pdf(monkeypatch, ["知识库正文"])
    env.pipeline.index_pdf("doc.pdf")

    answer = await env.pipeline.ask("问题？")
    assert answer.text == "标准答案[1]"
    assert len(answer.sources) == 1


@pytest.mark.anyio
async def test_ask_prompt_contains_numbered_context_and_question(env, monkeypatch):
    """prompt 必须携带编号上下文 + 问题；system 含引用指令。"""
    _patch_pdf(monkeypatch, ["知识库正文"])
    env.pipeline.index_pdf("doc.pdf")

    await env.pipeline.ask("核心问题？")

    messages = env.llm.chat.call_args.args[0]
    system, user = messages[0]["content"], messages[1]["content"]
    assert "标注你引用了哪段上下文" in system
    assert "[1]" in user
    assert "知识库正文" in user
    assert "核心问题？" in user


@pytest.mark.anyio
async def test_ask_returns_hint_when_store_empty(env):
    """空库：提示语 + 空 sources，且不调 LLM。"""
    answer = await env.pipeline.ask("问题？")
    assert answer.text == "知识库中没有找到相关内容"
    assert answer.sources == []
    env.llm.chat.assert_not_called()


@pytest.mark.anyio
async def test_ask_returns_empty_string_when_content_none(env, monkeypatch):
    """API 安全过滤返回 content=None：text 回退空串，sources 不受影响（检索先于生成）。"""
    _patch_pdf(monkeypatch, ["知识库正文"])
    env.pipeline.index_pdf("doc.pdf")
    env.llm.chat.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
    )

    answer = await env.pipeline.ask("问题？")
    assert answer.text == ""
    assert len(answer.sources) == 1


@pytest.mark.anyio
async def test_ask_sources_numbered_from_one_with_hit_data(env, monkeypatch):
    """citation 核心契约：number 从 1 递增，id/score/元数据来自命中的块。"""
    _patch_pdf(monkeypatch, ["第一页内容", "第二页内容"])
    env.pipeline.index_pdf("doc.pdf")

    answer = await env.pipeline.ask("问题？", top_k=5)

    assert [s.number for s in answer.sources] == [1, 2]
    assert [s.id for s in answer.sources] == ["doc.pdf:1:0", "doc.pdf:2:0"]
    for source in answer.sources:
        assert source.metadata["source"] == "doc.pdf"
        assert source.text == source.metadata["text"]
    assert answer.sources[0].score == 1.0


@pytest.mark.anyio
async def test_ask_sources_numbering_matches_prompt_context(env, monkeypatch):
    """prompt 里 [n] 的编号与 Answer.sources 的 number 一一对应（同一次检索同一编号）。"""
    _patch_pdf(monkeypatch, ["第一页内容", "第二页内容"])
    env.pipeline.index_pdf("doc.pdf")

    answer = await env.pipeline.ask("问题？", top_k=5)

    user = env.llm.chat.call_args.args[0][1]["content"]
    for source in answer.sources:
        assert f"[{source.number}] {source.text}" in user


# ---------- hybrid ----------

def test_index_pdf_feeds_keyword_index(hybrid_env, monkeypatch):
    """索引时关键词索引同步喂入：命中的块 id 与向量库同一套方案。"""
    _patch_pdf(monkeypatch, ["第一页内容", "第二页内容"])
    hybrid_env.pipeline.index_pdf("doc.pdf")

    hits = hybrid_env.keyword_index.search("第一页", top_k=5)

    assert [h.id for h in hits] == ["doc.pdf:1:0"]


def test_reindex_removes_old_chunks_from_keyword_index(hybrid_env, monkeypatch):
    """重复索引：关键词索引同样先删旧账（D20 stale-tail 的第二个表面）。"""
    _patch_pdf(monkeypatch, ["数据库是核心" * 100])
    hybrid_env.pipeline.index_pdf("doc.pdf")
    assert len(hybrid_env.keyword_index.search("数据库", top_k=50)) > 0

    _patch_pdf(monkeypatch, ["完全不相关内容"])
    hybrid_env.pipeline.index_pdf("doc.pdf")

    assert hybrid_env.keyword_index.search("数据库", top_k=50) == []


@pytest.mark.anyio
async def test_ask_fuses_keyword_hits_into_context(hybrid_env, monkeypatch):
    """dense 路漏掉的块，keyword 路命中后必须进入 prompt 上下文。"""
    _patch_pdf(monkeypatch, ["数据库是核心", "完全不相关"])
    hybrid_env.pipeline.index_pdf("doc.pdf")

    # dense 只返回不相关的块（模拟向量分低没进 top_k）
    original_search = hybrid_env.store.search
    monkeypatch.setattr(
        hybrid_env.store,
        "search",
        lambda query, top_k=5: [
            h for h in original_search(query, top_k) if h.id == "doc.pdf:2:0"
        ],
    )

    await hybrid_env.pipeline.ask("数据库", top_k=3)

    user = hybrid_env.llm.chat.call_args.args[0][1]["content"]
    assert "数据库是核心" in user  # keyword 路补上 dense 漏掉的块
    assert "完全不相关" in user  # dense 路自己的块也在


# ---------- rerank ----------

@pytest.mark.anyio
async def test_ask_two_stage_fetch_more_than_top_k(rerank_env, monkeypatch, mocker):
    """两段式：召回段按 fetch_k 取候选（> top_k），精排段截到 top_k。"""
    _patch_pdf(monkeypatch, ["第一页", "第二页"])
    rerank_env.pipeline.index_pdf("doc.pdf")

    spy = mocker.spy(rerank_env.store, "search")
    await rerank_env.pipeline.ask("问题？", top_k=1, fetch_k=5)

    assert spy.call_args.args[1] == 5  # 召回段用 fetch_k 而不是 top_k


@pytest.mark.anyio
async def test_ask_reranked_order_reaches_prompt(rerank_env, monkeypatch):
    """prompt 上下文顺序 = reranker 精排后的顺序（FakeReranker 按长度降序）。"""
    _patch_pdf(monkeypatch, ["短", "这段文本明显更长"])
    rerank_env.pipeline.index_pdf("doc.pdf")

    await rerank_env.pipeline.ask("问题？", top_k=2, fetch_k=5)

    user = rerank_env.llm.chat.call_args.args[0][1]["content"]
    assert user.index("这段文本明显更长") < user.index("短")


@pytest.mark.anyio
async def test_reranker_receives_fetch_candidates_and_top_k(rerank_env, monkeypatch, mocker):
    """reranker 收到召回段的全量候选，并被要求截到 top_k。"""
    _patch_pdf(monkeypatch, ["第一页", "第二页"])
    rerank_env.pipeline.index_pdf("doc.pdf")

    spy = mocker.spy(rerank_env.reranker, "rerank")
    await rerank_env.pipeline.ask("问题？", top_k=1, fetch_k=5)

    assert len(spy.call_args.args[1]) == 2  # 库里共 2 块，全量候选
    assert spy.call_args.args[2] == 1  # 精排目标 top_k


@pytest.mark.anyio
async def test_ask_without_reranker_fetches_top_k(env, monkeypatch, mocker):
    """无 reranker：fetch 退化为 top_k（旧行为不变）。"""
    _patch_pdf(monkeypatch, ["第一页", "第二页"])
    env.pipeline.index_pdf("doc.pdf")

    spy = mocker.spy(env.store, "search")
    await env.pipeline.ask("问题？", top_k=3)

    assert spy.call_args.args[1] == 3
