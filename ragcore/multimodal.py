"""CLIP-based cross-modal embeddings + SurrealDB ImageStore.

torch / open_clip / PIL are imported lazily inside ClipEmbedder._load() so that
``import ragcore.multimodal`` stays fast and the deterministic tests (which use
an injected fake model) run without those heavyweight dependencies being present.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from surrealdb import AsyncSurreal

from ragcore.config import SurrealConfig
from ragcore.errors import StoreError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure helpers (no torch required)
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def cross_modal_rank(
    query_vec: list[float],
    images: list[dict],
    k: int = 5,
) -> list[dict]:
    """Cosine-rank *images* against *query_vec*, return top-k.

    Each image dict must have an ``embedding`` key.  The returned dicts include
    ``id``, ``path``, and ``score``; ``embedding`` is dropped.
    """
    scored = []
    for img in images:
        emb = img.get("embedding", [])
        score = _cosine(query_vec, emb)
        row = {field: val for field, val in img.items() if field != "embedding"}
        row["score"] = score
        scored.append(row)
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:k]


# ---------------------------------------------------------------------------
# ClipEmbedder
# ---------------------------------------------------------------------------

class ClipEmbedder:
    """Wrap open_clip for text and image embedding.

    Parameters
    ----------
    model_name:  open_clip model architecture name (e.g. ``"ViT-B-32"``).
    pretrained:  pretrained weights tag (e.g. ``"laion2b_s34b_b79k"``).
    device:      ``"mps"``, ``"cuda"``, or ``"cpu"``.  ``"mps"`` falls back to
                 ``"cpu"`` when MPS is not available (with a logged warning).
    _model:      injection point for tests — when set, real torch is never loaded.
    """

    def __init__(
        self,
        model_name: str | None = None,
        pretrained: str | None = None,
        device: str | None = None,
        *,
        _model: Any = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._model = _model
        # Real model state — populated lazily by _load()
        self._clip_model = None
        self._preprocess = None
        self._tokenizer = None
        self._resolved_device: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        if self._model is not None:
            return list(self._model.encode_text(text))
        self._load()
        import torch  # noqa: PLC0415 — intentional lazy import
        with torch.no_grad():
            tokens = self._tokenizer([text]).to(self._resolved_device)
            features = self._clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            return features[0].cpu().tolist()

    def embed_image(self, path: str) -> list[float]:
        if self._model is not None:
            return list(self._model.encode_image(path))
        self._load()
        import torch  # noqa: PLC0415 — intentional lazy import
        from PIL import Image  # noqa: PLC0415 — intentional lazy import
        with torch.no_grad():
            img = Image.open(path).convert("RGB")
            tensor = self._preprocess(img).unsqueeze(0).to(self._resolved_device)
            features = self._clip_model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            return features[0].cpu().tolist()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazy-load torch + open_clip (called only on the real path)."""
        if self._clip_model is not None:
            return  # already loaded

        import open_clip  # noqa: PLC0415 — intentional lazy import
        import torch  # noqa: PLC0415 — intentional lazy import

        # Resolve device with MPS fallback
        requested = self.device or "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available — falling back to cpu")
            requested = "cpu"
        self._resolved_device = requested

        model_name = self.model_name or "ViT-B-32"
        pretrained = self.pretrained or "laion2b_s34b_b79k"

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        clip_model = clip_model.to(self._resolved_device)
        clip_model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        self._clip_model = clip_model
        self._preprocess = preprocess
        self._tokenizer = tokenizer


# ---------------------------------------------------------------------------
# ImageStore
# ---------------------------------------------------------------------------

class ImageStore:
    """SurrealDB table-class for image_embedding (mirrors JobQueue pattern)."""

    def __init__(self, config: SurrealConfig) -> None:
        self.config = config

    def _connect(self) -> AsyncSurreal:
        return AsyncSurreal(self.config.url)

    async def _signin(self, db: AsyncSurreal) -> None:
        await db.signin({"username": self.config.user, "password": self.config.password})
        await db.use(self.config.namespace, self.config.database)

    @staticmethod
    def _rows(result: Any) -> list:
        return result if isinstance(result, list) else (result or [])

    async def add_image(self, origin: str, path: str, embedding: list[float]) -> str:
        """Insert an image record; return the record id string."""
        db = self._connect()
        try:
            await self._signin(db)
            rows = self._rows(await db.query(
                "CREATE image_embedding SET origin=$o, path=$p, embedding=$e;",
                {"o": origin, "p": path, "e": embedding},
            ))
            return str(rows[0]["id"])
        except Exception as exc:
            raise StoreError(f"add_image failed: {exc}") from exc
        finally:
            await db.close()

    async def find_by_origin(self, origin: str) -> str | None:
        """Return the record id string if *origin* was already ingested, else None."""
        db = self._connect()
        try:
            await self._signin(db)
            rows = self._rows(await db.query(
                "SELECT id FROM image_embedding WHERE origin = $o LIMIT 1;", {"o": origin}))
            return str(rows[0]["id"]) if rows else None
        except Exception as e:
            raise StoreError(f"find_by_origin failed: {e}") from e
        finally:
            await db.close()

    async def search(self, query_vec: list[float], k: int = 5) -> list[dict]:
        """Return top-k image records by cosine similarity to *query_vec*."""
        db = self._connect()
        try:
            await self._signin(db)
            rows = self._rows(await db.query(
                "RETURN fn::image_search($q, $k);",
                {"q": query_vec, "k": k},
            ))
            return [
                {
                    "id": str(r["id"]),
                    "origin": r.get("origin"),
                    "path": r.get("path"),
                    "similarity": r.get("similarity"),
                }
                for r in rows
            ]
        except Exception as exc:
            raise StoreError(f"image search failed: {exc}") from exc
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# Top-level ingest helper
# ---------------------------------------------------------------------------

async def ingest_image(path: str, config: Any) -> tuple[str, bool]:
    """Embed *path* with CLIP and store in image_embedding.

    Returns ``(image_id, created)`` where *created* is False when the origin
    was already ingested (dedup pre-flight skips the expensive CLIP call).
    """
    store = ImageStore(config.surreal)
    existing = await store.find_by_origin(path)
    if existing is not None:
        return existing, False
    mm_cfg = config.multimodal
    embedder = ClipEmbedder(
        model_name=mm_cfg.model,
        pretrained=mm_cfg.pretrained,
        device=mm_cfg.device,
    )
    emb = embedder.embed_image(path)
    image_id = await store.add_image(origin=path, path=path, embedding=emb)
    return image_id, True
