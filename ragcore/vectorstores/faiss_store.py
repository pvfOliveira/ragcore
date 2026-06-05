"""FAISS-backed VectorStore adapter.

faiss-cpu and numpy are optional dependencies; they are lazy-imported inside
each method so the module can be imported without them installed (ImportError
is only raised at call time if a package is missing).

Cosine similarity is implemented via inner-product on L2-normalised vectors
(IndexFlatIP).  The sidecar ``meta.json`` stores ``[{"id": str, "text": str}]``
parallel to the FAISS row order so vectors can be mapped back to chunk data
without a second database.
"""
from __future__ import annotations

import json
import os


class FaissStore:
    """VectorStore implementation backed by an in-process FAISS IndexFlatIP.

    Args:
        path: Directory where ``index.faiss`` and ``meta.json`` are stored.
        dim:  Optional embedding dimension.  When *None* (default) the
              dimension is inferred from the first batch of embeddings passed
              to :meth:`add_embeddings`.
    """

    def __init__(self, path: str, dim: int | None = None) -> None:
        self._path = path
        self._dim = dim
        self._index = None  # faiss.IndexFlatIP, lazily created/loaded
        self._meta: list[dict] = []  # [{"id": str, "text": str}, ...]

        os.makedirs(path, exist_ok=True)

        index_file = self._index_file()
        meta_file = self._meta_file()
        if os.path.exists(index_file) and os.path.exists(meta_file):
            import faiss  # lazy import — optional extra

            self._index = faiss.read_index(index_file)
            with open(meta_file, encoding="utf-8") as fh:
                self._meta = json.load(fh)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_file(self) -> str:
        return os.path.join(self._path, "index.faiss")

    def _meta_file(self) -> str:
        return os.path.join(self._path, "meta.json")

    def _persist(self) -> None:
        import faiss  # lazy import — optional extra

        faiss.write_index(self._index, self._index_file())
        with open(self._meta_file(), "w", encoding="utf-8") as fh:
            json.dump(self._meta, fh)

    @staticmethod
    def _normalize(vec):  # type: ignore[return]
        """Return *vec* scaled to unit length (L2). Zero-vectors are unchanged."""
        import numpy as np  # lazy import — optional extra

        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec
        return vec / norm

    # ------------------------------------------------------------------
    # VectorStore protocol
    # ------------------------------------------------------------------

    async def add_embeddings(self, source_id: str, chunks: list[dict]) -> None:
        """Normalise and index *chunks*, then persist index + sidecar.

        Each chunk must contain ``{"id": str, "text": str, "embedding": list[float]}``.
        """
        import faiss  # lazy import — optional extra
        import numpy as np  # lazy import — optional extra

        if not chunks:
            return

        vecs = np.array([c["embedding"] for c in chunks], dtype="float32")
        # L2-normalise each row in place for cosine via inner product
        for i in range(len(vecs)):
            vecs[i] = self._normalize(vecs[i])

        dim = vecs.shape[1]

        if self._index is None:
            self._dim = dim
            self._index = faiss.IndexFlatIP(dim)

        self._index.add(vecs)
        self._meta.extend({"id": c["id"], "text": c["text"]} for c in chunks)
        self._persist()

    async def vector_search(
        self, query: list[float], k: int = 10
    ) -> list[dict]:
        """Return the *k* most similar chunks to *query* (higher score = more similar).

        Returns:
            List of ``{"id": str, "text": str, "score": float}`` dicts,
            ordered from highest score (most similar) to lowest.
            Returns ``[]`` when the index is empty or not yet created.
        """
        import numpy as np  # lazy import — optional extra

        if self._index is None or self._index.ntotal == 0:
            return []

        q = np.array([query], dtype="float32")
        q[0] = self._normalize(q[0])

        # Clamp k to the number of indexed vectors to avoid -1 padding
        effective_k = min(k, self._index.ntotal)
        scores, idxs = self._index.search(q, effective_k)

        results = []
        for j, idx in enumerate(idxs[0]):
            if idx == -1:
                continue
            meta = self._meta[int(idx)]
            results.append(
                {"id": meta["id"], "text": meta["text"], "score": float(scores[0][j])}
            )
        return results
