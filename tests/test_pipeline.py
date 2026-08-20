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
    """FakeVectorStore + fake embedder + mock LLM。"""
    store = FakeVectorStore()
    llm = mocker.Mock()
    llm.chat.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="标准答案[1]"))]
    )
    pipeline = create_rag_pipeline(FakeEmbedder(), store, llm)
    return SimpleNamespace(pipeline=pipeline, store=store, llm=llm)


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


# ---------- ask ----------

def test_ask_returns_llm_text(env, monkeypatch):
    """LLM 的 content 原样返回。"""
    _patch_pdf(monkeypatch, ["知识库正文"])
    env.pipeline.index_pdf("doc.pdf")

    assert env.pipeline.ask("问题？") == "标准答案[1]"


def test_ask_prompt_contains_numbered_context_and_question(env, monkeypatch):
    """prompt 必须携带编号上下文 + 问题；system 含引用指令。"""
    _patch_pdf(monkeypatch, ["知识库正文"])
    env.pipeline.index_pdf("doc.pdf")

    env.pipeline.ask("核心问题？")

    messages = env.llm.chat.call_args.args[0]
    system, user = messages[0]["content"], messages[1]["content"]
    assert "标注你引用了哪段上下文" in system
    assert "[1]" in user
    assert "知识库正文" in user
    assert "核心问题？" in user


def test_ask_returns_hint_when_store_empty(env):
    """空库：返回提示语，且不调 LLM。"""
    assert env.pipeline.ask("问题？") == "知识库中没有找到相关内容"
    env.llm.chat.assert_not_called()


def test_ask_returns_empty_string_when_content_none(env, monkeypatch, mocker):
    """API 安全过滤返回 content=None：ask 返回空串而不是 None。"""
    _patch_pdf(monkeypatch, ["知识库正文"])
    env.pipeline.index_pdf("doc.pdf")
    env.llm.chat.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
    )

    assert env.pipeline.ask("问题？") == ""
