"""D21: BM25 关键词索引 + RRF 融合测试。

纯 Python 实现，无重依赖，CI 直接运行——BM25 是确定性数学，
用真实实现测（不 fake），测试本身就是公式接对与否的证明。
"""

from rag.hybrid import BM25Index, _tokenize, rrf_fuse
from rag.vector_store import SearchHit

# ---------- tokenize ----------

def test_tokenize_ascii_words_and_chinese_bigrams():
    """英文按词提取，中文按二元组切分。"""
    assert _tokenize("RAG 向量检索") == ["RAG", "向量", "量检", "检索"]


def test_tokenize_skips_punctuation():
    """标点和空白不产生 token。"""
    assert _tokenize("Hello, 世界！") == ["Hello", "世界"]


def test_tokenize_empty_string():
    assert _tokenize("") == []


# ---------- BM25Index ----------

def _make_index(docs: dict[str, str]) -> BM25Index:
    index = BM25Index()
    for doc_id, text in docs.items():
        index.add(doc_id, text)
    return index


def test_search_returns_only_matching_doc():
    """只含查询词的文档被命中；未匹配的文档不得以 0 分混入。"""
    index = _make_index({
        "d1": "向量检索 是 RAG 的核心技术",
        "d2": "python 是编程语言",
        "d3": "数据库索引优化",
    })

    hits = index.search("数据库", top_k=3)

    assert [h.id for h in hits] == ["d3"]


def test_search_ranks_multiple_matches_by_relevance():
    """词频高的文档排前面。"""
    index = _make_index({
        "d1": "数据库索引优化",
        "d2": "数据库 数据库 数据库 慢查询 数据库优化",
    })

    hits = index.search("数据库", top_k=3)

    assert hits[0].id == "d2"


def test_search_no_match_returns_empty():
    """无任何 token 命中返回 []，而不是 0 分文档。"""
    index = _make_index({"d1": "向量检索 是 RAG 的核心技术"})

    assert index.search("完全无关词") == []


def test_search_empty_index_returns_empty():
    assert BM25Index().search("任何") == []


def test_search_matches_english_words():
    index = _make_index({
        "d1": "python is a language",
        "d2": "rust is a language",
    })

    assert [h.id for h in index.search("python", top_k=5)] == ["d1"]


def test_search_top_k_truncates():
    index = _make_index({f"d{i}": f"数据库 知识 {i}" for i in range(5)})

    assert len(index.search("数据库", top_k=3)) == 3


def test_search_hits_carry_text_metadata():
    """命中的 metadata 必须带 {"text": 原文}——融合后进 prompt 的契约。"""
    index = _make_index({"d1": "向量检索 是 RAG 的核心技术"})

    hits = index.search("向量")

    assert hits[0].metadata == {"text": "向量检索 是 RAG 的核心技术"}


def test_delete_removes_doc_and_rebuilds_stats():
    """删除后 df/avgdl 从剩余文档重建：删掉的词再查为空。"""
    index = _make_index({"d1": "数据库索引", "d2": "编程语言"})
    index.delete(["d1"])

    assert index.search("数据库") == []
    assert [h.id for h in index.search("编程")] == ["d2"]


# ---------- rrf_fuse ----------

def _hit(id_: str, score: float = 1.0, metadata: dict | None = None) -> SearchHit:
    return SearchHit(id=id_, score=score, metadata=metadata or {"text": id_})


def test_rrf_fuse_ranks_doc_in_both_lists_first():
    """两边榜单都出现的文档融合分最高，排第一。"""
    dense = [_hit("a"), _hit("b"), _hit("c")]
    sparse = [_hit("b"), _hit("d"), _hit("c")]

    fused = rrf_fuse(dense, sparse, top_k=5)

    assert [h.id for h in fused] == ["b", "c", "a", "d"]


def test_rrf_fuse_includes_docs_from_single_list():
    """只出现在一边榜单的文档也进融合榜（hybrid 的意义）。"""
    dense = [_hit("a"), _hit("b")]
    sparse = [_hit("b"), _hit("c")]

    fused = rrf_fuse(dense, sparse, top_k=5)

    assert {h.id for h in fused} == {"a", "b", "c"}
    assert fused[0].id == "b"


def test_rrf_fuse_truncates_to_top_k():
    fused = rrf_fuse([_hit(f"a{i}") for i in range(10)], [], top_k=3)

    assert len(fused) == 3


def test_rrf_fuse_uses_rank_not_original_score():
    """融合只看榜单内排名，不看原始分数：score 9.9 但排第 2 的，输给 score 0.1 排第 1 的。"""
    dense = [_hit("x"), _hit("a", score=9.9)]
    sparse = [_hit("b", score=0.1)]

    fused = rrf_fuse(dense, sparse, top_k=3)

    assert fused[-1].id == "a"
    assert "b" in [h.id for h in fused[:2]]


def test_rrf_fuse_prefers_dense_metadata_on_collision():
    """同 id 两边都命中时，取 dense 的元数据（更丰富：source/page/text）。"""
    dense = [_hit("a", metadata={"text": "dense", "page": 1})]
    sparse = [_hit("a", metadata={"text": "sparse"})]

    fused = rrf_fuse(dense, sparse, top_k=1)

    assert fused[0].metadata == {"text": "dense", "page": 1}
