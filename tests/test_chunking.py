"""D20: chunking 测试。

纯逻辑、无外部依赖（rag 包重依赖已懒加载，import 安全），CI 直接运行。

核心不变量：步长 = chunk_size - overlap，
块 i 的文本 == text[i*step : i*step + chunk_size]（无缝隙、无漂移）。
"""
import pytest

from rag.chunking import chunk_text


def test_chunks_match_source_positions():
    """最强不变量：每块内容等于源文本按步长滑窗切片。"""
    text = "0123456789abcdefghij" * 7
    size, overlap = 25, 8

    chunks = chunk_text(text, size, overlap)

    step = size - overlap
    for i, c in enumerate(chunks):
        assert c.text == text[i * step : i * step + size]


def test_adjacent_chunks_overlap():
    """相邻块共享 overlap 长度的边界区——overlap 存在的意义。"""
    text = "abcdefghij" * 10  # 100 字，步长 20

    chunks = chunk_text(text, chunk_size=30, overlap=10)

    assert len(chunks) == 5  # 起始位置 0, 20, 40, 60, 80
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.text[-10:] == b.text[:10]


def test_local_id_and_index_increment():
    """id 是块内序号，chunk_index 连续递增（pipeline 据此铸全局 id）。"""
    chunks = chunk_text("x" * 100, chunk_size=30, overlap=5)

    assert [c.id for c in chunks] == [str(i) for i in range(len(chunks))]
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_short_text_yields_single_chunk():
    """文本短于 chunk_size：整段成一块，不补零。"""
    chunks = chunk_text("短文本", chunk_size=500, overlap=100)

    assert len(chunks) == 1
    assert chunks[0].text == "短文本"


def test_last_chunk_may_be_shorter():
    """尾部不补零：最后一块可能短于 chunk_size。"""
    text = "x" * 33  # 步长 6：起始 0, 6, ..., 30，最后一块 3 字

    chunks = chunk_text(text, chunk_size=10, overlap=4)

    assert len(chunks[-1].text) == 3


def test_empty_text_returns_empty_list():
    assert chunk_text("") == []


def test_overlap_equal_chunk_size_rejected():
    """overlap >= chunk_size 时步长 <= 0（死循环），必须拒绝。"""
    with pytest.raises(ValueError, match="larger"):
        chunk_text("abc", chunk_size=10, overlap=10)
