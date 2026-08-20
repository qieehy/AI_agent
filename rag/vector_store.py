from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float
    metadata: dict


class VectorStore(ABC):
    @abstractmethod
    def add(self, ids: list[str], vectors: list[list[float]], metadata: list[dict] | None = None) -> None: ...
    @abstractmethod
    def search(self, query: list[float], top_k: int=5) -> list[SearchHit]: ...
    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...
    @abstractmethod
    def update(self, ids: list[str], vectors: list[list[float]], metadata: list[dict] | None = None) -> None: ...

class FAISSVectorStore(VectorStore):
    def __init__(self):
        if faiss is None:
            raise ImportError("FAISSVectorStore 需要 rag extras: pip install -e '.[rag]'")
        self._index: faiss.IndexFlatIP | None = None
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._metadata: list[dict] = []

    def add(self, ids, vectors, metadata = None) -> None:
        if not ids:
            return
        if len(ids) != len(vectors):
            raise ValueError(f"vectors 数量必须和 ids 一致: {len(vectors)} vs {len(ids)}")

        if metadata is None:
            metadata = [{} for _ in ids]

        if len(metadata) != len(ids):
            raise ValueError(
                f"metadata 数量必须和 ids 一致: {len(metadata)} vs {len(ids)}"
            )

        if len(ids) != len(set(ids)):
            raise ValueError(
                "id 不能重复"
            )

        if set(ids) & set(self._ids):
            raise ValueError("id 已存在，如需覆盖请用 update()")

        vectors_np = np.array(
            vectors,
            dtype=np.float32
        )

        if vectors_np.ndim != 2:
            raise ValueError(
                "vectors 必须是二维数组"
            )

        if self._index is None:
            self._index = faiss.IndexFlatIP(vectors_np.shape[1])

        elif vectors_np.shape[1] != self._index.d:
            raise ValueError(f"期望{self._index.d}维, 得到{vectors_np.shape[1]}维")

        self._index.add(vectors_np)
        self._ids.extend(ids)
        self._vectors.extend(vectors_np.tolist())
        self._metadata.extend(metadata)


    def search(self, query: list[float], top_k: int=5) -> list[SearchHit]:
        """按向量相似度检索，返回按分数降序的 SearchHit 列表。空库返回 []。"""
        if self._index is None or self._index.ntotal == 0:
            return []
        distances, indices = self._index.search(
            np.array([query], dtype=np.float32), k=top_k
        )

        hits = []
        for i, pos in enumerate(indices[0]):
            if pos == -1:
                continue
            hits.append(SearchHit(
                id = self._ids[pos],
                score = float(distances[0][i]),
                metadata = self._metadata[pos]
            ))
        return hits

    def delete(self, ids: list[str]) -> None:
        """删除向量。flat index 不支持原位删除，这里用重建实现：过滤平行列表后新建同维度索引。
            O(N)，教学/小规模（<1 万向量）正确且简单；大规模需换 IndexIDMap2.remove_ids 或墓碑标记。"""
        if not self._ids:
            return
        to_delete = set(ids)
        keep = [i for i, id_ in enumerate(self._ids) if id_ not in to_delete]

        if len(keep) == len(self._ids):
            return

        self._ids = [self._ids[i] for i in keep]
        self._vectors = [self._vectors[i] for i in keep]
        self._metadata = [self._metadata[i] for i in keep]

        if not self._ids:
            self._index = None
            return
        self._index = faiss.IndexFlatIP(self._index.d)
        self._index.add(np.array(self._vectors, dtype=np.float32))

    def update(self, ids: list[str], vectors: list[list[float]],
               metadata: list[dict] | None = None) -> None:
        """更新向量 = delete + add。注意：非原子操作，两次重建之间 id 短暂不存在。"""
        self.delete(ids)
        self.add(ids, vectors, metadata)
