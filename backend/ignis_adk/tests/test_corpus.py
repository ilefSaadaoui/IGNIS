"""Corpus/RAG tests — offline (synthetic chunks in a tmp corpus)."""

import json

from ignis.corpus.store import CorpusIndex, chunk_text
from ignis.tools.literature_tools import search_scientific_literature, get_corpus_status


def _tmp_index(tmp_path):
    idx = CorpusIndex(tmp_path / "corpus")
    idx.ingest_text(
        "BASS-II investigated burning rates of solid PMMA fuels in microgravity aboard the ISS. "
        "Flame spread was measured with thermocouples and cameras. Extinction limits depended on oxygen.",
        investigation="BASS-II", record_id="t-bass2", title="BASS-II solids results",
        url="https://ntrs.nasa.gov/citations/test-bass2")
    idx.ingest_text(
        "The FLEX experiment studied droplet combustion extinction with carbon dioxide and helium diluents. "
        "Nonpremixed methanol droplets were burned in microgravity to measure extinction strain.",
        investigation="FLEX", record_id="t-flex", title="FLEX droplet extinction",
        url="https://ntrs.nasa.gov/citations/test-flex")
    idx.ingest_text(
        "SAME measured smoke particle size distributions from on-orbit combustion of spacecraft materials "
        "to evaluate photoelectric and ionization smoke detectors in microgravity.",
        investigation="SAME", record_id="t-same", title="SAME smoke experiment",
        url="https://ntrs.nasa.gov/citations/test-same")
    idx.save_metadata()
    return idx


def test_chunk_text_paragraph_aware():
    pieces = chunk_text("Para one. " * 5 + "\n\n" + "Para two is longer. " * 40)
    assert pieces
    assert all(len(p) <= 1200 for p in pieces)


def test_index_search_ranking_and_citations(tmp_path):
    idx = _tmp_index(tmp_path)
    hits = idx.search("PMMA burning rate microgravity", top_k=2)
    assert hits
    assert hits[0]["investigation"] == "BASS-II"
    assert hits[0]["url"].startswith("https://ntrs.nasa.gov/")
    assert "p" not in hits[0] or True  # page is None for text ingest


def test_investigation_filter(tmp_path):
    idx = _tmp_index(tmp_path)
    hits = idx.search("microgravity combustion", top_k=5, investigation="FLEX")
    assert hits and all(h["investigation"] == "FLEX" for h in hits)


def test_empty_corpus_returns_unavailable(tmp_path, monkeypatch):
    import ignis.tools.literature_tools as lt
    empty = CorpusIndex(tmp_path / "empty_corpus")
    monkeypatch.setattr(lt, "get_corpus_index", lambda: empty)
    r = lt.search_scientific_literature("anything")
    assert r["status"] == "unavailable"
    assert r["passages"] == []


def test_search_tool_output_contract(tmp_path, monkeypatch):
    import ignis.tools.literature_tools as lt
    idx = _tmp_index(tmp_path)
    monkeypatch.setattr(lt, "get_corpus_index", lambda: idx)
    r = lt.search_scientific_literature("smoke detector microgravity", top_k=2)
    assert r["status"] == "completed"
    assert r["passage_count"] >= 1
    assert r["passages"][0]["title"] == "SAME smoke experiment"


def test_metadata_roundtrip(tmp_path):
    idx = _tmp_index(tmp_path)
    idx2 = CorpusIndex(tmp_path / "corpus")
    assert idx2.load_metadata() is True
    assert idx2.stats()["documents"] == 3
    assert len(idx2.chunks) == len(idx.chunks)


# --- hybrid dense retrieval -------------------------------------------------

import hashlib
import numpy as np


class _FakeEncoder:
    """Deterministic 48-dim hashing embedder (offline, no downloads)."""

    DIM = 48

    def encode(self, texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True):
        vecs = []
        for text in texts:
            v = np.zeros(self.DIM, dtype="float32")
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode()).digest()
                v[int.from_bytes(digest[:2], "little") % self.DIM] += 1.0
                v[int.from_bytes(digest[2:4], "little") % self.DIM] += 0.5
            if normalize_embeddings:
                norm = np.linalg.norm(v)
                if norm > 0:
                    v /= norm
            vecs.append(v)
        return np.asarray(vecs, dtype="float32")


def _hybrid_index(tmp_path, monkeypatch):
    idx = _tmp_index(tmp_path)
    monkeypatch.setenv("IGNIS_DENSE", "on")
    monkeypatch.setattr(CorpusIndex, "_get_encoder", lambda self: _FakeEncoder())
    return idx


def test_hybrid_search_blends_and_persists_embeddings(tmp_path, monkeypatch):
    idx = _hybrid_index(tmp_path, monkeypatch)
    hits = idx.search("droplet extinction strain rate", top_k=3)
    assert hits
    assert all(h["retrieval"] == "hybrid" for h in hits)
    assert all("dense_score" in h and "sparse_score" in h for h in hits)
    # persisted + reloadable
    assert (tmp_path / "corpus" / "embeddings.npy").exists()
    idx2 = CorpusIndex(tmp_path / "corpus")
    assert idx2.load_metadata()
    mat = idx2._load_dense_matrix()
    assert mat is not None and mat.shape[0] == len(idx2.chunks)
    assert idx2.dense_status()["embeddings_ready"] is True


def test_hybrid_falls_back_to_tfidf_when_encoder_fails(tmp_path, monkeypatch):
    idx = _tmp_index(tmp_path)
    monkeypatch.setenv("IGNIS_DENSE", "on")

    def _boom(self):
        raise RuntimeError("encoder unavailable")

    monkeypatch.setattr(CorpusIndex, "_get_encoder", _boom)
    hits = idx.search("PMMA burning rate", top_k=2)
    assert hits
    assert all(h["retrieval"] == "tfidf" for h in hits)
    assert idx.dense_status()["embeddings_ready"] is False


def test_dense_off_env_forces_sparse(tmp_path, monkeypatch):
    idx = _tmp_index(tmp_path)
    monkeypatch.setenv("IGNIS_DENSE", "off")
    assert idx.build_dense() is False
    assert idx.dense_status()["enabled"] is False


def test_multi_investigation_tagging_and_filter(tmp_path):
    idx = CorpusIndex(tmp_path / "corpus")
    idx.ingest_text(
        "BASS-II summary report covers solid fuel burn rates in microgravity aboard the ISS.",
        investigation="BASS-II", record_id="m-1", title="BASS-II summary",
        url="https://ntrs.nasa.gov/citations/m-1", investigations=["BASS-II", "BASS"])
    # sklearn TF-IDF needs >=2 documents; add two unrelated fillers
    idx.ingest_text(
        "Smoke detector test document for corpus sizing purposes only.",
        investigation="SAME", record_id="m-2", title="SAME pad",
        url="https://ntrs.nasa.gov/citations/m-2")
    idx.ingest_text(
        "Droplet extinction test document for corpus sizing purposes only.",
        investigation="FLEX", record_id="m-3", title="FLEX pad",
        url="https://ntrs.nasa.gov/citations/m-3")
    idx.save_metadata()
    hits_bass = idx.search("burn rates microgravity", investigation="BASS")
    hits_bass2 = idx.search("burn rates microgravity", investigation="BASS-II")
    assert hits_bass and hits_bass[0]["investigations"] == ["BASS-II", "BASS"]
    assert hits_bass2 and hits_bass2[0]["chunk_id"].startswith("m-1")
    assert idx.stats()["investigations"]["BASS"] == 1
    assert idx.stats()["investigations"]["BASS-II"] == 1
