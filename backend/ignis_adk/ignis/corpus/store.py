"""Local scientific-literature corpus with hybrid retrieval.

Retrieval strategy:
- Sparse leg: TF-IDF (1-2 grams, cosine) — deterministic, dependency-light,
  works fully offline.
- Dense leg (optional): sentence-transformers (default all-MiniLM-L6-v2,
  384-dim, CPU-friendly). Embeddings are persisted next to the metadata
  (embeddings.npy + embeddings_meta.json) so re-ingestion re-encodes
  automatically when the chunk list changes.
- Fusion: per-query min-max normalisation of both score vectors, then a
  weighted blend (IGNIS_DENSE_WEIGHT, default 0.6 dense / 0.4 sparse).

Set IGNIS_DENSE=off to force sparse-only retrieval (used by tests). IGNIS_DENSE=auto
(default) falls back to TF-IDF with a logged warning when the encoder or its
weights are unavailable — the system never silently pretends to be dense.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("ignis.corpus.store")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150

DENSE_MODEL = os.getenv("IGNIS_DENSE_MODEL", "all-MiniLM-L6-v2")


def _dense_mode() -> str:
    return os.getenv("IGNIS_DENSE", "auto").strip().lower()


def _dense_weight() -> float:
    try:
        w = float(os.getenv("IGNIS_DENSE_WEIGHT", "0.6"))
    except ValueError:
        w = 0.6
    return max(0.0, min(1.0, w))


_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "were", "was", "are", "of",
    "to", "in", "on", "a", "an", "what", "which", "who", "whom", "how", "why", "when",
    "where", "does", "did", "is", "it", "its", "as", "by", "at", "or", "be", "been",
}


def _query_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9][a-z0-9.\-]{1,}", text.lower()) if len(t) >= 3 and t not in _STOPWORDS}


def _title_overlap(qtokens: set[str], title: str) -> float:
    if not qtokens:
        return 0.0
    ttoks = _query_tokens(title or "")
    if not ttoks:
        return 0.0
    return len(qtokens & ttoks) / len(qtokens)


def _minmax(vec):
    import numpy as np

    arr = np.asarray(vec, dtype="float32")
    span = float(arr.max() - arr.min())
    if span <= 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.min()) / span


@dataclass
class CorpusChunk:
    chunk_id: str
    investigation: str
    title: str
    url: str
    page: int | None
    text: str
    investigations: list[str] | None = None  # all tags this chunk belongs to (multi-investigation docs)

    @property
    def tags(self) -> list[str]:
        return self.investigations or [self.investigation]


@dataclass
class CorpusDocument:
    investigation: str
    record_id: str
    title: str
    url: str
    published: str | None
    pages: int
    abstract: str | None = None
    investigations: list[str] | None = None

    @property
    def tags(self) -> list[str]:
        return self.investigations or [self.investigation]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "doc"


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware chunking with overlap; never mid-word if avoidable."""
    if not text.strip():
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|(?<=\.)\s{2,}", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= chunk_chars:
            buf = f"{buf} {para}".strip()
            continue
        if buf:
            chunks.append(buf)
        if len(para) <= chunk_chars:
            buf = para
            continue
        # very long paragraph: hard-split with overlap
        step = max(1, chunk_chars - overlap)
        for start in range(0, len(para), step):
            piece = para[start:start + chunk_chars].strip()
            if piece:
                chunks.append(piece)
        buf = ""
    if buf:
        chunks.append(buf)
    return chunks


class CorpusIndex:
    """In-memory TF-IDF index over ingested chunks; persisted metadata on disk."""

    def __init__(self, corpus_dir: Path | str | None = None) -> None:
        self.corpus_dir = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS_DIR
        self.metadata_path = self.corpus_dir / "metadata.json"
        self.documents: list[CorpusDocument] = []
        self.chunks: list[CorpusChunk] = []
        self._vectorizer = None
        self._matrix = None
        self._loaded = False
        self._encoder = None
        self._dense_matrix = None
        self._dense_checked = False

    # --- ingestion ----------------------------------------------------------

    def ingest_pdf(self, pdf_path: Path | str, *, investigation: str, record_id: str,
                   title: str, url: str, published: str | None = None,
                   abstract: str | None = None,
                   investigations: list[str] | None = None) -> dict[str, Any]:
        from pypdf import PdfReader

        p = Path(pdf_path)
        if not p.exists() or p.stat().st_size == 0:
            return {"status": "failed", "errors": [f"missing/empty PDF: {p}"], "chunks": 0}
        if p.read_bytes()[:5] != b"%PDF-":
            return {"status": "failed", "errors": [f"not a PDF file (magic bytes mismatch): {p.name}"], "chunks": 0}
        try:
            reader = PdfReader(str(p))
            page_texts = [(i + 1, (page.extract_text() or "").strip()) for i, page in enumerate(reader.pages)]
        except Exception as exc:
            return {"status": "failed", "errors": [f"PDF parse failed for {p.name}: {exc}"], "chunks": 0}

        tags = list(dict.fromkeys(investigations or [investigation]))
        added = 0
        for page_no, page_text in page_texts:
            for piece in chunk_text(page_text):
                self.chunks.append(CorpusChunk(
                    chunk_id=f"{record_id}-p{page_no}-{added}",
                    investigation=investigation,
                    title=title,
                    url=url,
                    page=page_no,
                    text=piece,
                    investigations=list(tags),
                ))
                added += 1
        self.documents.append(CorpusDocument(
            investigation=investigation, record_id=record_id, title=title, url=url,
            published=published, pages=len(page_texts), abstract=(abstract or "")[:2000] or None,
            investigations=list(tags),
        ))
        self._loaded = False
        return {"status": "completed", "record_id": record_id, "investigation": investigation,
                "title": title, "pages": len(page_texts), "chunks": added, "url": url}

    def ingest_text(self, text: str, *, investigation: str, record_id: str,
                    title: str, url: str, published: str | None = None,
                    investigations: list[str] | None = None) -> dict[str, Any]:
        """Ingest raw text (used by tests and for non-PDF STI)."""
        tags = list(dict.fromkeys(investigations or [investigation]))
        added = 0
        for piece in chunk_text(text):
            self.chunks.append(CorpusChunk(
                chunk_id=f"{record_id}-t-{added}", investigation=investigation,
                title=title, url=url, page=None, text=piece, investigations=list(tags),
            ))
            added += 1
        self.documents.append(CorpusDocument(
            investigation=investigation, record_id=record_id, title=title,
            url=url, published=published, pages=0, investigations=list(tags),
        ))
        self._loaded = False
        return {"status": "completed", "record_id": record_id, "chunks": added, "url": url}

    # --- persistence of metadata --------------------------------------------

    def save_metadata(self) -> None:
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self._dense_checked = False
        self._dense_matrix = None
        payload = {
            "documents": [d.__dict__ for d in self.documents],
            "chunk_count": len(self.chunks),
            "chunks": [c.__dict__ for c in self.chunks],
        }
        self.metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    def load_metadata(self) -> bool:
        if not self.metadata_path.exists():
            return False
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.documents = [CorpusDocument(**d) for d in payload.get("documents", [])]
            for d in self.documents:
                if not d.investigations:
                    d.investigations = [d.investigation]
            self.chunks = [CorpusChunk(**c) for c in payload.get("chunks", [])]
            for c in self.chunks:
                if not c.investigations:
                    c.investigations = [c.investigation]
            self._loaded = False
            self._dense_checked = False
            self._dense_matrix = None
            return True
        except Exception as exc:
            logger.warning("corpus metadata load failed: %s", exc)
            return False

    # --- dense retrieval machinery -------------------------------------------

    def _embeddings_paths(self) -> tuple[Path, Path]:
        return self.corpus_dir / "embeddings.npy", self.corpus_dir / "embeddings_meta.json"

    def _chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    def _load_dense_matrix(self):
        import numpy as np

        npy, meta = self._embeddings_paths()
        if not npy.exists() or not meta.exists():
            return None
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
            if payload.get("model") != DENSE_MODEL:
                return None
            if payload.get("chunk_ids") != self._chunk_ids():
                return None
            mat = np.load(npy)
            if mat.shape[0] != len(self.chunks):
                return None
            return mat.astype("float32")
        except Exception as exc:
            logger.warning("dense embeddings load failed (%s); will re-encode", exc)
            return None

    def _save_dense_matrix(self, mat) -> None:
        import numpy as np

        npy, meta = self._embeddings_paths()
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy, mat.astype("float32"))
        meta.write_text(
            json.dumps({"model": DENSE_MODEL, "chunk_ids": self._chunk_ids()}, ensure_ascii=False),
            encoding="utf-8",
        )

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(DENSE_MODEL)
        return self._encoder

    def build_dense(self, force: bool = False) -> bool:
        """Build (or load) the dense embedding matrix. Returns True when ready."""
        if _dense_mode() == "off":
            self._dense_checked, self._dense_matrix = True, None
            return False
        if self._dense_checked and self._dense_matrix is not None and not force:
            return True
        self._ensure_sparse()
        if not self.chunks:
            self._dense_checked, self._dense_matrix = True, None
            return False
        if not force:
            mat = self._load_dense_matrix()
            if mat is not None:
                self._dense_matrix, self._dense_checked = mat, True
                return True
        try:
            import numpy as np

            encoder = self._get_encoder()
            embs = encoder.encode(
                [c.text for c in self.chunks],
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            mat = np.asarray(embs, dtype="float32")
            self._save_dense_matrix(mat)
            self._dense_matrix = mat
            logger.info("dense embeddings ready: %s chunks, model=%s", mat.shape[0], DENSE_MODEL)
        except Exception as exc:
            logger.warning("dense encoder unavailable (%s); falling back to TF-IDF only", exc)
            self._dense_matrix = None
        self._dense_checked = True
        return self._dense_matrix is not None

    def dense_status(self) -> dict:
        """Cheap, encoder-free status (safe for status tools and stats())."""
        if not self.chunks and self.metadata_path.exists():
            self.load_metadata()
        if _dense_mode() == "off":
            return {"enabled": False, "reason": "IGNIS_DENSE=off", "model": DENSE_MODEL}
        npy, meta = self._embeddings_paths()
        ready = False
        if npy.exists() and meta.exists():
            try:
                payload = json.loads(meta.read_text(encoding="utf-8"))
                ready = payload.get("model") == DENSE_MODEL and payload.get("chunk_ids") == self._chunk_ids()
            except Exception:
                ready = False
        if self._dense_matrix is not None:
            ready = True
        return {"enabled": True, "model": DENSE_MODEL, "embeddings_ready": ready,
                "weight": _dense_weight()}

    # --- retrieval ------------------------------------------------------------

    def _ensure_sparse(self) -> None:
        if self._loaded:
            return
        if not self.chunks:
            if not self.load_metadata():
                self._loaded = True
                return
        if not self.chunks:
            self._loaded = True
            return
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            stop_words="english",
        )
        self._matrix = self._vectorizer.fit_transform([c.text for c in self.chunks])
        self._loaded = True

    def _ensure_index(self) -> None:
        self._ensure_sparse()

    def search(self, query: str, *, top_k: int = 5, investigation: str | None = None) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            return []
        self._ensure_index()
        if self._vectorizer is None or self._matrix is None or not self.chunks:
            return []
        import numpy as np
        from sklearn.metrics.pairwise import linear_kernel

        qv = self._vectorizer.transform([query.strip()])
        sparse = linear_kernel(qv, self._matrix).ravel().astype("float32")

        dense = None
        if self.build_dense() and self._dense_matrix is not None:
            try:
                qv_dense = self._get_encoder().encode(
                    [query.strip()], normalize_embeddings=True, show_progress_bar=False,
                )
                dense = (np.asarray(qv_dense, dtype="float32") @ self._dense_matrix.T).ravel()
            except Exception as exc:
                logger.warning("dense query encoding failed (%s); using TF-IDF only", exc)
                dense = None

        qtokens = _query_tokens(query)
        title_boost = np.array([_title_overlap(qtokens, c.title) for c in self.chunks], dtype="float32")
        if dense is not None:
            scores = (1.0 - _dense_weight()) * _minmax(sparse) + _dense_weight() * _minmax(dense)
            scores = scores + 0.15 * title_boost
            retrieval = "hybrid"
        else:
            scores = sparse + 0.15 * title_boost
            dense = np.zeros_like(sparse)
            retrieval = "tfidf"

        order = scores.argsort()[::-1]
        inv = (investigation or "").strip().lower() or None
        results: list[dict[str, Any]] = []
        try:
            top_k = max(1, min(top_k, 20))
        except (TypeError, ValueError):
            top_k = 5
        for idx in order:
            chunk = self.chunks[idx]
            if inv and inv not in [t.lower() for t in chunk.tags]:
                continue
            score = float(scores[idx])
            if retrieval == "tfidf" and score <= 0.0:
                break
            results.append({
                "chunk_id": chunk.chunk_id,
                "investigation": chunk.investigation,
                "title": chunk.title,
                "url": chunk.url,
                "page": chunk.page,
                "score": round(score, 4),
                "sparse_score": round(float(sparse[idx]), 4),
                "dense_score": round(float(dense[idx]), 4),
                "retrieval": retrieval,
                "investigations": chunk.tags,
                "text": chunk.text,
            })
            if len(results) >= top_k:
                break
        return results

    def stats(self) -> dict[str, Any]:
        self._ensure_index()
        by_inv: dict[str, int] = {}
        for d in self.documents:
            for tag in d.tags:
                by_inv[tag] = by_inv.get(tag, 0) + 1
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "investigations": by_inv,
            "corpus_dir": str(self.corpus_dir),
            "dense": self.dense_status(),
        }


_index: CorpusIndex | None = None


def get_corpus_index(corpus_dir: Path | str | None = None) -> CorpusIndex:
    global _index
    if corpus_dir is not None:
        idx = CorpusIndex(corpus_dir)
        idx.load_metadata()
        idx._ensure_index()
        return idx
    if _index is None:
        _index = CorpusIndex()
        _index.load_metadata()
        _index._ensure_index()
    return _index
