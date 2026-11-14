"""Download verified public NASA NTRS PDFs into the local IGNIS corpus.

Usage:
    python scripts/ingest_corpus.py              # ingest manifest PDFs
    python scripts/ingest_corpus.py --refresh    # re-discover IDs via NTRS search, then ingest
    python scripts/ingest_corpus.py --max-per-investigation 3

Provenance is recorded in data/corpus/metadata.json (title, NTRS URL, page).
Files that fail magic-byte validation are rejected, never parsed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ignis.corpus.manifest import MANIFEST, NTRS_CITATION, NTRS_RECORD, NTRS_SEARCH, USER_AGENT  # noqa: E402
from ignis.corpus.store import CorpusIndex, _dense_mode  # noqa: E402

logger = logging.getLogger("ignis.ingest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _get_json(url: str, **params):
    resp = requests.get(url, params=params or None, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def refresh_manifest(max_per: int) -> dict[str, list[int]]:
    from ignis.corpus.manifest import INVESTIGATION_QUERIES

    manifest: dict[str, list[int]] = {}
    for inv, query in INVESTIGATION_QUERIES.items():
        try:
            payload = _get_json(NTRS_SEARCH, q=query, **{"page.size": max_per * 2})
        except Exception as exc:
            logger.warning("refresh failed for %s: %s", inv, exc)
            continue
        ids = [int(r["id"]) for r in payload.get("results", []) if r.get("id")]
        manifest[inv] = ids[:max_per]
        logger.info("refresh %s -> %s", inv, manifest[inv])
    return manifest


def record_details(record_id: int) -> dict:
    return _get_json(NTRS_RECORD.format(id=record_id))


def pick_pdf_link(details: dict) -> str | None:
    for dl in details.get("downloads", []) or []:
        if (dl.get("mimetype") or "").lower() == "application/pdf":
            links = dl.get("links") or {}
            return links.get("pdf") or links.get("original") or links.get("fulltext")
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-discover IDs from NTRS search first")
    parser.add_argument("--max-per-investigation", type=int, default=3)
    parser.add_argument("--corpus-dir", default=None)
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else Path(__file__).resolve().parents[1] / "data" / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh:
        discovered = refresh_manifest(args.max_per_investigation)
        # Union: keep hand-verified base IDs, add newly-discovered ones, dedupe, keep cap.
        manifest = {}
        for inv in dict.fromkeys(list(MANIFEST) + list(discovered)):
            merged = list(dict.fromkeys(list(MANIFEST.get(inv, [])) + list(discovered.get(inv, []))))
            manifest[inv] = merged[: args.max_per_investigation]
    else:
        manifest = MANIFEST

    # Some NTRS records match several investigation queries. Ingest each record once
    # and tag it with ALL investigations it was found under (a doc relevant to BASS
    # and BASS-II should answer both filters). Primary (display) tag prefers the
    # most specific label.
    SPECIFICITY = {"BASS-II": 0, "SAME-R": 0, "FLEX": 1, "ACME": 1, "CIR": 1, "FLAME-SPREAD": 1,
                   "FLAMMABILITY": 1, "COOL-FLAMES": 1, "SMOKE-DET": 1, "MAT-FLAM": 1,
                   "BASS": 2, "SAME": 2, "SAFFIRE": 2, "FIRE-SAFETY": 2}
    order = {inv: i for i, inv in enumerate(manifest)}
    buckets: dict[int, list[str]] = {}
    for investigation, ids in manifest.items():
        for rid in ids[: args.max_per_investigation]:
            buckets.setdefault(int(rid), [])
            if investigation not in buckets[int(rid)]:
                buckets[int(rid)].append(investigation)

    def _sort_tags(tags: list[str]) -> list[str]:
        return sorted(tags, key=lambda t: (SPECIFICITY.get(t, 3), order.get(t, 99)))

    index = CorpusIndex(corpus_dir)
    report = {"downloaded": [], "skipped": [], "failed": [], "record_investigations": {}}
    for rid, tags in buckets.items():
        tags = _sort_tags(tags)
        primary = tags[0]
        report["record_investigations"][str(rid)] = tags
        investigation = primary
        dest = corpus_dir / f"{rid}.pdf"
        try:
            details = record_details(rid)
        except Exception as exc:
            report["failed"].append({"id": rid, "error": f"metadata fetch failed: {exc}"})
            continue
        pdf_link = pick_pdf_link(details)
        title = (details.get("title") or f"NTRS record {rid}").strip()
        published = details.get("distributionDate")
        abstract = (details.get("abstract") or "").strip()
        if not pdf_link:
            report["skipped"].append({"id": rid, "investigation": ",".join(tags), "title": title,
                                      "reason": "no public PDF download", "url": NTRS_CITATION.format(id=rid)})
            continue
        url = f"https://ntrs.nasa.gov{pdf_link}" if pdf_link.startswith("/") else pdf_link
        try:
            if not dest.exists():
                resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
                resp.raise_for_status()
                if resp.content[:5] != b"%PDF-":
                    report["skipped"].append({"id": rid, "investigation": ",".join(tags), "title": title,
                                              "reason": "download was not a PDF", "url": NTRS_CITATION.format(id=rid)})
                    continue
                dest.write_bytes(resp.content)
                time.sleep(0.5)  # be polite to NTRS
            result = index.ingest_pdf(
                dest, investigation=primary, record_id=str(rid),
                title=title, url=NTRS_CITATION.format(id=rid), published=published, abstract=abstract,
                investigations=tags,
            )
            if result["status"] == "completed":
                report["downloaded"].append(result)
                logger.info("ingested %s [%s] (%s pages, %s chunks)", rid, ",".join(tags), result["pages"], result["chunks"])
            else:
                report["failed"].append({"id": rid, "error": result.get("errors")})
        except Exception as exc:
            report["failed"].append({"id": rid, "error": f"{exc.__class__.__name__}: {exc}"})
            logger.warning("failed %s: %s", rid, exc)

    index.save_metadata()
    if _dense_mode() != "off":
        logger.info("building dense embeddings (this can take a few minutes on CPU) ...")
        report["dense_embeddings"] = index.build_dense()
    else:
        report["dense_embeddings"] = False
    summary = {
        "corpus_dir": str(corpus_dir),
        "documents": len(index.documents),
        "chunks": len(index.chunks),
        "downloaded": report["downloaded"],
        "skipped": report["skipped"],
        "failed": report["failed"],
        "record_investigations": report["record_investigations"],
        "dense_embeddings": report.get("dense_embeddings"),
    }
    (corpus_dir / "ingest_report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in summary.items()}, indent=1))
    return 0 if index.documents else 1


if __name__ == "__main__":
    raise SystemExit(main())
