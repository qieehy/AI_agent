"""D19: Vector Store 测试。

策略：
- `importorskip`：faiss 未安装时自动跳过（CI 只装 dev extras）
- 合成向量（dim=4 且归一化）测检索机制：快、确定性，不加载 BGE 模型
- 语义质量已在 D18 测过；这里只测"存进去能找到、删了能消失、改了能反映"
- 函数级 fixture：每个测试一个全新空库，互不污染
"""
from __future__ import annotations

import pytest

pytest.importorskip("faiss")

from rag import create_vector_store

# 合成向量：已归一化（|v|=1），内积 = 余弦相似度，方便直觉验证
APPLE = [1.0, 0.0, 0.0, 0.0]   # 苹果
FRUIT = [0.8, 0.6, 0.0, 0.0]   # 水果（离苹果近）
CAR = [0.0, 0.0, 1.0, 0.0]     # 汽车（离苹果远）


@pytest.fixture
def store():
    """函数级：每个测试一个全新空库。"""
    return create_vector_store()


# ---------- CRUD ----------

def test_add_then_search_hits(store):
    """存进去能按相似度找回来，id 和 metadata 完整传回。"""
    store.add(["apple", "fruit", "car"],
              [APPLE, FRUIT, CAR],
              [{"kind": "fruit"}, {"kind": "fruit"}, {"kind": "vehicle"}])

    hits = store.search(APPLE, top_k=3)

    assert [h.id for h in hits] == ["apple", "fruit", "car"]
    assert hits[0].metadata == {"kind": "fruit"}


def test_scores_descending(store):
    """返回分数降序，且数值与手算余弦一致。"""
    store.add(["apple", "car"], [APPLE, CAR])

    hits = store.search(APPLE, top_k=2)

    assert hits[0].score == pytest.approx(1.0)      # 自己 vs 自己
    assert hits[1].score == pytest.approx(0.0)      # 正交向量
    assert hits[0].score >= hits[1].score


def test_related_closer_than_unrelated(store):
    """验收核心：语义相近的排在无关的前面。"""
    store.add(["apple", "fruit", "car"], [APPLE, FRUIT, CAR])

    ids = [h.id for h in store.search(APPLE, top_k=3)]

    assert ids.index("fruit") < ids.index("car")


def test_delete_removes(store):
    """删除后检索不到，且不干扰其他条目。"""
    store.add(["apple", "car"], [APPLE, CAR])
    store.delete(["apple"])

    hits = store.search(APPLE, top_k=2)

    assert [h.id for h in hits] == ["car"]


def test_delete_nonexistent_is_noop(store):
    """删除不存在的 id：幂等，不报错不误删。"""
    store.add(["apple"], [APPLE])
    store.delete(["ghost"])

    assert [h.id for h in store.search(APPLE, top_k=3)] == ["apple"]


def test_delete_all_returns_to_empty(store):
    """删光后回到空库状态，且能继续写入。"""
    store.add(["apple"], [APPLE])
    store.delete(["apple"])

    assert store.search(APPLE) == []

    store.add(["car"], [CAR])
    assert [h.id for h in store.search(CAR, top_k=3)] == ["car"]


def test_update_changes_vector(store):
    """update 后旧方向不再相似，新方向命中，metadata 同步更新。"""
    store.add(["apple"], [APPLE])
    store.update(["apple"], [CAR], [{"kind": "vehicle"}])

    hits = store.search(CAR, top_k=3)
    assert [h.id for h in hits] == ["apple"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].metadata == {"kind": "vehicle"}

    assert store.search(APPLE, top_k=3)[0].score == pytest.approx(0.0)


def test_update_adds_if_missing(store):
    """update 天然支持 insert-or-update：id 不存在时等于 add。"""
    store.update(["brand_new"], [CAR])

    assert [h.id for h in store.search(CAR, top_k=3)] == ["brand_new"]


# ---------- 边界 ----------

def test_search_empty_store(store):
    """空库查询返回 []，不崩溃。"""
    assert store.search(APPLE) == []


def test_top_k_exceeds_size(store):
    """库里向量少于 top_k：只返回实际数量，无 -1 哨兵泄漏。"""
    store.add(["only"], [APPLE])

    hits = store.search(APPLE, top_k=5)

    assert len(hits) == 1
    assert hits[0].id == "only"


def test_add_empty_is_noop(store):
    """空批次 add：无操作，库保持空。"""
    store.add([], [], [])

    assert store.search(APPLE) == []


# ---------- 校验 ----------

def test_add_rejects_duplicate_ids_in_batch(store):
    """同批次重复 id 拒绝。"""
    with pytest.raises(ValueError, match="重复"):
        store.add(["a", "a"], [APPLE, CAR])


def test_add_rejects_existing_id(store):
    """已存在的 id 拒绝，提示用 update()。"""
    store.add(["a"], [APPLE])

    with pytest.raises(ValueError, match="update"):
        store.add(["a"], [CAR])


def test_add_rejects_dimension_mismatch(store):
    """第二次 add 维度不一致，报错带双方维度。"""
    store.add(["a"], [APPLE])

    with pytest.raises(ValueError, match="期望4维"):
        store.add(["b"], [[1.0, 0.0, 0.0]])


def test_add_rejects_length_mismatch(store):
    """ids 和 vectors 长度不一致拒绝。"""
    with pytest.raises(ValueError, match="数量"):
        store.add(["a", "b"], [APPLE])


def test_add_rejects_metadata_length_mismatch(store):
    """metadatas 长度不一致拒绝。"""
    with pytest.raises(ValueError, match="metadata"):
        store.add(["a"], [APPLE], [{"x": 1}, {"y": 2}])
