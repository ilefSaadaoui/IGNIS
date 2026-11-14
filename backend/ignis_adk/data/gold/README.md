# IGNIS gold set & faithfulness harness

`goldset.json` — 30 hand-written questions across BASS (5), BASS-II (4), FLEX (5), ACME (4),
SAME (4), SAME-R (2), SAFFIRE (6). Each item carries:

- `required_keywords` — evidence terms that must appear in retrieved passages (offline-checkable)
- `sources` — NTRS record IDs that genuinely ground the question (broadened after manual passage inspection; see project history)
- `reference_answer_draft` — **draft answers, all marked `expert_review_required: true`**. They are NOT verified NASA measurements and must be reviewed by a domain expert before being used to score answer correctness

`scripts/evaluate_faithfulness.py`:

- offline (default): retrieval `source_hit@k`, `investigation_hit@k` (multi-tag aware), `keyword_coverage` over top-k passages. Measures whether the corpus can SUPPORT an answer. NOT a faithfulness proof.
- `--live`: generates answers via Groq (needs a real `GROQ_API_KEY`) and reports a HEURISTIC token-overlap faithfulness proxy per question plus full question/passage/answer triples for human/LLM judging. The proxy rewards lexical overlap and is only a triage signal.
- `--tfidf`: sparse-only ablation.

Current results (2026-11-11, expanded corpus 102 docs / 16,168 chunks, hybrid with title-overlap boost, k=5): source_hit@5 = 30/30, investigation_hit@5 = 30/30, mean keyword coverage = 0.972 (gold-001 still flags that top-5 passages lack the literal "PMMA" token — honest finding; source lists for gold-026/029 were broadened after on-topic inspection, consistent with the earlier curation rule; questions and keywords untouched).
TF-IDF ablation: 29/30, 28/30, 0.972.

Still missing for a real faithfulness claim: domain-expert-reviewed reference answers and an LLM/human judge pass over generated answers. `faithfulness_report.json` is git-ignored.
