from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """切块结果。id 是块内序号；全局唯一 id（{source}:{page}:{chunk_index}）由 RAGPipeline 铸造。"""

    id: str
    text: str
    metadata: dict


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    """固定大小 + overlap 切分，步长 = chunk_size - overlap。

    overlap 的意义：固定切块必然切在句子中间，半句话的向量语义残缺；
    重叠区保证任何完整句子至少完整落在某一块里。

    - chunk_size <= overlap 抛 ValueError；空文本返回 []
    - 最后一块可能短于 chunk_size（尾部不补零）
    """
    if chunk_size <= overlap:
        raise ValueError("Chunk size must be larger than overlap")
    if not text:
        return []

    step = chunk_size - overlap
    start = 0
    index = 0
    chunks = []

    while start < len(text):
        end = start + chunk_size

        piece = text[start: end]
        chunks.append(
            Chunk(
                id=str(index),
                text = piece,
                metadata = {
                    "chunk_index": index,
                }
            )
        )
        index += 1
        start += step

    return chunks
