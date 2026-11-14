"""Retrieval-quality evaluation harness (offline, deterministic).

Each case: query -> expected investigation + required keywords in the top
passage. This measures whether the corpus actually surfaces the right paper —
it is NOT a measure of answer faithfulness (that needs a judge model).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ignis.corpus.store import get_corpus_index  # noqa: E402

CASES = [
    {"query": "Burning and Suppression of Solids BASS-II PMMA burning rate", "expect_investigation": "BASS-II",
     "keywords": ["burn", "pmma"]},
    {"query": "flame extinguishment experiment extinction droplet", "expect_investigation": "FLEX",
     "keywords": ["extinction", "droplet", "flame"]},
    {"query": "smoke aerosol measurement experiment particle size detector", "expect_investigation": "SAME",
     "keywords": ["smoke", "particle"]},
    {"query": "SAFFIRE spacecraft fire experiment large scale", "expect_investigation": "SAFFIRE",
     "keywords": ["fire", "spacecraft", "saffire"]},
    {"query": "Advanced Combustion via Microgravity Experiments ACME flame", "expect_investigation": "ACME",
     "keywords": ["combustion", "flame", "acme"]},
]


def _match(hits, case):
    """Pass if a passage from the expected investigation ranks in top-3 and contains >=1 keyword."""
    for rank, h in enumerate(hits[:3], start=1):
        if h["investigation"] != case["expect_investigation"]:
            continue
        blob = (h["text"] + " " + h["title"]).lower()
        if any(k.lower() in blob for k in case["keywords"]):
            return rank, h
    return None, hits[0] if hits else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfidf", action="store_true", help="force sparse-only retrieval (IGNIS_DENSE=off)")
    args = parser.parse_args()
    if args.tfidf:
        import os
        os.environ["IGNIS_DENSE"] = "off"

    index = get_corpus_index()
    stats = index.stats()
    dense = (stats.get("dense") or {})
    print(f"Corpus: {stats['documents']} documents, {stats['chunks']} chunks")
    print(f"Dense: {dense}")
    top1 = 0
    hit3 = 0
    for case in CASES:
        hits = index.search(case["query"], top_k=5)
        if not hits:
            print(f"FAIL {case['expect_investigation']:8s} | no passages")
            continue
        rank, match = _match(hits, case)
        ok = rank is not None
        top1 += int(hits[0]["investigation"] == case["expect_investigation"])
        hit3 += int(ok)
        detail = f"rank={rank} score={match['score']:.3f}" if ok else f"top={hits[0]['investigation']} score={hits[0]['score']:.3f}"
        print(f"{'PASS' if ok else 'FAIL'} {case['expect_investigation']:8s} | {detail} | {hits[0]['title'][:60]}")
    print(f"\nhit@1={top1}/{len(CASES)}  hit@3(investigation+keyword)={hit3}/{len(CASES)}")
    return 0 if hit3 == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
