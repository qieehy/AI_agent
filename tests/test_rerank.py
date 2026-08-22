"""D21: CrossEncoderReranker 真模型测试。

策略（延续 D18 test_embeddings 模式）：
- importorskip：sentence_transformers 未安装时跳过（CI 只装 dev extras）
- module 级 fixture：2.3GB 模型只加载一次（CPU 首次加载约半分钟）
- 测相对排序与结构性质，不测具体分数值（logits 随模型版本变化）
"""
from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers")

from rag import create_reranker
from rag.vector_store import SearchHit


@pytest.fixture(scope="module")
def reranker():
    """module 级：整个测试文件共享一个模型实例。"""
    return create_reranker()


def _hit(id_: str, text: str) -> SearchHit:
    return SearchHit(id=id_, score=1.0, metadata={"text": text, "source": "doc.pdf", "page": 1})


def test_rerank_orders_relevant_first(reranker):
    """验收核心：与问题相关的块排到无关块前面。"""
    query = "向量数据库是什么"
    hits = [
        _hit("unrelated", "今天天气不错，适合出门散步"),
        _hit("relevant", "向量数据库是专门用于存储和检索向量数据的系统"),
    ]

    ranked = reranker.rerank(query, hits, top_k=2)

    assert [h.id for h in ranked] == ["relevant", "unrelated"]


def test_rerank_preserves_id_and_metadata(reranker):
    """reranker 是纯重排器：id 与 metadata 原样保留，只有 score 被精排分替换。"""
    ranked = reranker.rerank("向量数据库", [_hit("a", "向量数据库是存储向量的系统")], top_k=1)

    assert ranked[0].id == "a"
    assert ranked[0].metadata == {"text": "向量数据库是存储向量的系统", "source": "doc.pdf", "page": 1}
    assert isinstance(ranked[0].score, float)


def test_rerank_truncates_to_top_k(reranker):
    hits = [_hit(str(i), f"向量数据库的知识点 {i}") for i in range(4)]

    ranked = reranker.rerank("向量数据库", hits, top_k=2)

    assert len(ranked) == 2


def test_rerank_empty_hits_returns_empty(reranker):
    assert reranker.rerank("问题", [], top_k=3) == []
