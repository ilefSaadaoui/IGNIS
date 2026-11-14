"""Gold-set evidence-coverage + answer-faithfulness harness.

Two layers, kept strictly separate:

1. Offline (default) — NO LLM required.
   For each gold question: retrieve top-k passages, then measure
     - source_hit@k : an expected source record_id appears in the top-k chunks
     - investigation_hit@k : at least one top-k passage is from the expected investigation
     - keyword_coverage    : fraction of required evidence keywords present in the top-k text
   This measures whether the corpus can SUPPORT a correct answer. It is not a
   judgement about any generated answer.

2. --live — needs GROQ_API_KEY (real key; placeholder will fail).
   Generates one answer per question with the configured Groq model using ONLY the
   retrieved passages, then computes a HEURISTIC faithfulness proxy: the fraction of
   the answer's content tokens (alpha, len>=4, stopwords removed) that also appear in
   the retrieved passages. This proxy rewards lexical overlap and is NOT proof of
   faithfulness or correctness — treat it as a triage signal and have a human/LLM
   judge score the printed question/passage/answer triples for real sign-off.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ignis.corpus.store import get_corpus_index  # noqa: E402

GOLD_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "goldset.json"
REPORT_PATH = Path(__file__).resolve().parents[1] / "data" / "gold" / "faithfulness_report.json"

_STOP = {
    "that", "this", "with", "from", "into", "they", "their", "there", "were", "been",
    "have", "has", "had", "and", "the", "are", "was", "for", "you", "your", "can",
    "about", "into", "these", "those", "which", "while", "where", "when", "what",
}


def _record_id_of(chunk: dict) -> str:
    return str(chunk.get("chunk_id", "")).split("-")[0]


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z][a-z0-9\-]{2,}", text.lower()) if t not in _STOP and len(t) >= 4}


def _score_offline(index, item: dict, k: int) -> dict:
    hits = index.search(item["question"], top_k=k)
    rid_hits = [h for h in hits if _record_id_of(h) in set(item.get("sources", []))]
    want_inv = item["investigation"].lower()
    inv_hits = [
        h for h in hits
        if want_inv in [t.lower() for t in (h.get("investigations") or [h.get("investigation", "")])]
    ]
    blob = " ".join(f"{h.get('title', '')} {h.get('text', '')}" for h in hits).lower()
    keywords = [kw.lower() for kw in item.get("required_keywords", [])]
    covered = [kw for kw in keywords if kw in blob]
    return {
        "id": item["id"],
        "investigation": item["investigation"],
        "question": item["question"],
        "top": [{"chunk_id": h["chunk_id"], "title": h["title"], "investigation": h["investigation"],
                 "score": h.get("score"), "page": h.get("page"), "url": h.get("url")} for h in hits],
        "source_hit": bool(rid_hits),
        "investigation_hit": bool(inv_hits),
        "keyword_coverage": round(len(covered) / len(keywords), 4) if keywords else None,
        "keywords_missing": [kw for kw in keywords if kw not in covered],
        "_hits_text": blob,
    }


def _live_answer(question: str, passages: list[dict], model: str) -> str:
    import litellm

    context = "\n\n".join(
        f"[{i + 1}] {h.get('title')} (p.{h.get('page')})\n{h.get('text', '')}" for i, h in enumerate(passages)
    )
    prompt = (
        "Answer the question using ONLY the passages below. Cite passage numbers like [1]. "
        "If the passages do not contain the answer, say so explicitly.\n\n"
        f"Passages:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    resp = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
    return (resp.choices[0].message.content or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="also generate Groq answers + heuristic faithfulness")
    parser.add_argument("--tfidf", action="store_true", help="force sparse-only retrieval")
    parser.add_argument("--k", type=int, default=5, help="top-k passages per question")
    args = parser.parse_args()
    if args.tfidf:
        import os
        os.environ["IGNIS_DENSE"] = "off"

    if not GOLD_PATH.exists():
        print(f"gold set not found: {GOLD_PATH}", file=sys.stderr)
        return 2
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    index = get_corpus_index()
    stats = index.stats()
    print(f"Gold set: {len(gold)} questions | Corpus: {stats['documents']} docs / {stats['chunks']} chunks | k={args.k}")

    results = [_score_offline(index, item, args.k) for item in gold]

    if args.live:
        from ignis.config import get_settings
        cfg = get_settings()
        if not cfg.groq_api_key_configured:
            print("\n--live requested but GROQ_API_KEY is missing/placeholder. Set a real key in .env.\n"
                  "Offline results above are still valid; no answers were fabricated.", file=sys.stderr)
            return 2
        model = cfg.GROQ_MODEL
        for r in results:
            passages = [h for h in index.search(r["question"], top_k=args.k)]
            try:
                answer = _live_answer(r["question"], passages, model)
            except Exception as exc:
                answer = f"[generation failed: {exc.__class__.__name__}: {exc}]"
            toks = _content_tokens(answer)
            passage_toks = _content_tokens(r["_hits_text"])
            supported = len(toks & passage_toks) / len(toks) if toks else None
            r["answer"] = answer
            r["faithfulness_proxy"] = round(supported, 4) if supported is not None else None
            r["faithfulness_note"] = ("HEURISTIC token-overlap proxy; not proof of faithfulness — "
                                      "needs human/LLM judge sign-off.")

    source_hits = sum(r["source_hit"] for r in results)
    inv_hits = sum(r["investigation_hit"] for r in results)
    covs = [r["keyword_coverage"] for r in results if r["keyword_coverage"] is not None]
    print(f"source_hit@{args.k}: {source_hits}/{len(results)}   "
          f"investigation_hit@{args.k}: {inv_hits}/{len(results)}   "
          f"mean keyword coverage: {sum(covs) / len(covs):.3f}")
    for r in results:
        cov = r["keyword_coverage"]
        flag = "OK " if (r["source_hit"] and (cov or 0) >= 0.5) else "LOW"
        extra = f" proxy={r.get('faithfulness_proxy')}" if "faithfulness_proxy" in r else ""
        print(f"[{flag}] {r['id']:9s} {r['investigation']:8s} src={r['source_hit']} cov={cov}{extra} :: {r['question'][:64]}")
        if r["keywords_missing"]:
            print(f"        missing keywords: {r['keywords_missing']}")

    for r in results:
        r.pop("_hits_text", None)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "live" if args.live else "offline",
        "k": args.k,
        "retrieval": "tfidf" if args.tfidf else "auto (hybrid when available)",
        "corpus": {k: v for k, v in stats.items() if k != "dense"} | {"dense": stats.get("dense")},
        "summary": {
            "questions": len(results),
            f"source_hit@{args.k}": source_hits,
            f"investigation_hit@{args.k}": inv_hits,
            "mean_keyword_coverage": round(sum(covs) / len(covs), 4) if covs else None,
            "mean_faithfulness_proxy": (round(sum(r["faithfulness_proxy"] for r in results if r.get("faithfulness_proxy") is not None) /
                                              max(1, len([r for r in results if r.get("faithfulness_proxy") is not None])), 4)
                                        if args.live else None),
        },
        "questions": results,
        "honesty_notes": [
            "Offline metrics measure evidence RETRIEVAL, not answer faithfulness.",
            "--live faithfulness_proxy is token-overlap heuristic; a human or LLM judge must score the",
            "printed question/passage/answer triples before any faithfulness claim is made.",
            "Gold answers are drafts marked expert_review_required; no numeric NASA measurements are asserted",
            "that are not present in the retrieved passages.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
