"""Scientific literature retrieval from the local IGNIS corpus (RAG).

Every passage returned carries title, NTRS URL and page so answers can cite
the exact source. If the corpus is empty or the query matches nothing, the
tool says so — it never invents a passage.
"""

from __future__ import annotations

from typing import Any

from ignis.corpus.store import get_corpus_index


def search_scientific_literature(
    query: str,
    top_k: int = 5,
    investigation: str | None = None,
) -> dict[str, Any]:
    """Search the local corpus of NASA combustion papers.

    Args:
        query: Natural-language query, e.g. "BASS-II burning rate PMMA oxygen".
        top_k: Max passages (1-20).
        investigation: Optional filter, e.g. 'BASS-II', 'FLEX', 'ACME', 'SAME', 'SAFFIRE'.
    """
    if not isinstance(query, str) or not query.strip():
        return {"status": "failed", "errors": ["query must be a non-empty string"], "passages": []}
    index = get_corpus_index()
    stats = index.stats()
    if stats["documents"] == 0:
        return {
            "status": "unavailable",
            "errors": ["Corpus is empty. Run: python scripts/ingest_corpus.py (requires network)."],
            "passages": [],
            "corpus": stats,
        }
    passages = index.search(query, top_k=top_k, investigation=investigation)
    dense = (stats.get("dense") or {})
    retrieval = "hybrid (TF-IDF + dense)" if dense.get("embeddings_ready") else "TF-IDF (dense unavailable)"
    return {
        "status": "completed" if passages else "partial",
        "query": query.strip(),
        "investigation_filter": investigation,
        "passages": passages,
        "passage_count": len(passages),
        "corpus": stats,
        "retrieval": retrieval,
        "limitations": [
            f"Retrieval: {retrieval}; read the passages before citing — lexical/embedding match is not evidence.",
            "Only documents ingested via scripts/ingest_corpus.py are searchable.",
        ],
    }


def get_corpus_status() -> dict[str, Any]:
    """Report corpus size and per-investigation coverage."""
    return {"status": "completed", "corpus": get_corpus_index().stats()}
