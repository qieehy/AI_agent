import math
import re
from collections import Counter

from .vector_store import SearchHit


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []

    for match in re.finditer(
        r"[A-Za-z0-9]+|[\u4e00-\u9fff]+",
        text,
    ):
        part = match.group()

        if re.fullmatch(r"[A-Za-z0-9]+", part):
            tokens.append(part)
        else:
            for i in range(len(part) - 1):
                tokens.append(part[i : i + 2])

    return tokens


class BM25Index:
    K1 = 1.5
    B = 0.75
    def __init__(self) -> None:
        self._docs: dict[str, str] = {}
        self._term_freqs: dict[str, Counter[str]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._metadata: dict[str, dict] = {}


    def add(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        tokens = _tokenize(text)
        term_freq = Counter(tokens)

        self._docs[doc_id] = text
        self._term_freqs[doc_id] = term_freq
        self._doc_lengths[doc_id] = len(tokens)
        self._metadata[doc_id] = {**(metadata or {}), "text": text}


    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not self._docs:
            return []

        query_tokens = _tokenize(query)

        n = len(self._docs)
        avgdl = sum(self._doc_lengths.values()) / n

        scores: dict[str, float] = {}

        for doc_id, term_freq in self._term_freqs.items():
            doc_length = self._doc_lengths[doc_id]
            score = 0.0

            for token in query_tokens:
                freq = term_freq.get(token, 0)

                if freq == 0:
                    continue

                df = sum(
                    token in tf
                    for tf in self._term_freqs.values()
                )

                idf = math.log(
                    (n - df + 0.5) / (df + 0.5) + 1
                )

                numerator = freq * (self.K1 + 1)

                denominator = (
                        freq
                        + self.K1
                        * (
                                1
                                - self.B
                                + self.B * doc_length / avgdl
                        )
                )

                score += idf * numerator / denominator

            # 无任何查询 token 命中的文档 0 分，不得混入结果（否则污染 RRF 融合榜）
            if score == 0:
                continue
            scores[doc_id] = score

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            SearchHit(
                id=doc_id,
                score=score,
                metadata=self._metadata[doc_id],
            )
            for doc_id, score in ranked[:top_k]
        ]

    def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            self._docs.pop(doc_id, None)
            self._term_freqs.pop(doc_id, None)
            self._doc_lengths.pop(doc_id, None)
            self._metadata.pop(doc_id, None)


def rrf_fuse(
        dense_hits: list[SearchHit],
        sparse_hits: list[SearchHit],
        top_k: int,
        k: int = 60,
) -> list[SearchHit]:
    rrf_scores: dict[str, float] = {}
    hits_by_id: dict[str, SearchHit] = {}

    for rank, hit in enumerate(dense_hits):
        hits_by_id[hit.id] = hit

        score = 1 / (k + rank + 1)

        rrf_scores[hit.id] = (
                rrf_scores.get(hit.id, 0.0)
                + score
        )

    for rank, hit in enumerate(sparse_hits):
        # 同 id 两边都命中时 dense 优先：dense 元数据更丰富（source/page/text）
        if hit.id not in hits_by_id:
            hits_by_id[hit.id] = hit

        score = 1 / (k + rank + 1)

        rrf_scores[hit.id] = (
                rrf_scores.get(hit.id, 0.0)
                + score
        )

    ranked_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )

    return [
        SearchHit(
            id=doc_id,
            score=rrf_scores[doc_id],
            metadata=hits_by_id[doc_id].metadata,
        )
        for doc_id in ranked_ids[:top_k]
    ]
