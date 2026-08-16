"""D18: Embedding 服务测试。

策略：
- `importorskip`：sentence_transformers 未安装时自动跳过（CI 只装 dev extras）
- module 级 fixture：模型只加载一次，避免每个测试重新加载（首次运行会从 HuggingFace 下载 ~100MB）
- 测结构与语义性质，不测具体数值（向量值随模型版本微调而变）
"""
from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers")

from rag import create_embedding_service

BGE_SMALL_ZH_DIM = 512  # BAAI/bge-small-zh-v1.5 输出维度


@pytest.fixture(scope="module")
def service():
    """module 级：整个测试文件共享一个模型实例。"""
    return create_embedding_service()


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度 = 点积（向量已归一化，模长为 1）。"""
    return sum(x * y for x, y in zip(a, b, strict=True))


# ---------- 结构测试 ----------

def test_embed_returns_fixed_dimension(service):
    """单条文本 → 512 维 float 列表。"""
    v = service.embed("这是一个测试句子")

    assert isinstance(v, list)
    assert len(v) == BGE_SMALL_ZH_DIM
    assert all(isinstance(x, float) for x in v)


def test_embed_normalized(service):
    """normalize_embeddings=True → 向量模长 ≈ 1。"""
    v = service.embed("归一化验证")

    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-5


def test_embed_batch_matches_single(service):
    """embed_batch 结果与逐条 embed 一致。"""
    texts = ["今天天气很好", "Python 装饰器怎么用", "向量数据库是什么"]

    batch = service.embed_batch(texts)
    single = [service.embed(t) for t in texts]

    assert len(batch) == len(texts)
    for b, s in zip(batch, single, strict=True):
        assert len(b) == len(s) == BGE_SMALL_ZH_DIM
        for x, y in zip(b, s, strict=True):
            assert abs(x - y) < 1e-5


def test_embed_batch_empty(service):
    """空列表 → 空列表。"""
    assert service.embed_batch([]) == []


# ---------- 语义性质测试 ----------

def test_similar_texts_closer_than_unrelated(service):
    """验收核心：语义相近的句子，向量距离更近。"""
    a = service.embed("今天天气很好")
    b = service.embed("今日阳光明媚")
    c = service.embed("Python 装饰器怎么用")

    sim_ab = _cosine(a, b)   # 相似句
    sim_ac = _cosine(a, c)   # 无关句

    assert sim_ab > sim_ac, f"相似度应更高: {sim_ab} vs {sim_ac}"
