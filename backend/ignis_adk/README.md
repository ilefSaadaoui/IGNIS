# IGNIS — Space Combustion Intelligence

**Team:** RaoSpace Labs · **Lead:** Ilef Saadaoui · **Challenge:** *Flame in Freefall: AI-Powered Fire Safety Insights from Microgravity Combustion Data*

IGNIS is an evidence-grounded, multi-agent scientific research system that turns NASA
microgravity combustion research into searchable, comparable, explainable knowledge.
Built on **Google Agent Development Kit (ADK) 2.14.0** with **Groq** inference via ADK's
**LiteLLM** integration. No Gemini API is used anywhere.

> Core principle: every scientific conclusion must be supported by evidence (now including
> retrieved, citable passages from a real local NASA corpus), transparent deterministic
> calculation, or clearly-labelled AI interpretation. Nothing is fabricated.

For custom frontend integration, see [FRONTEND_DOCUMENTATION.md](./FRONTEND_DOCUMENTATION.md).

---

## 1. What works today (verified)

| Capability | Status |
|---|---|
| Google ADK orchestration (root + 8 specialists) | ✅ implemented & tested |
| Groq via ADK `LiteLlm` (`groq/...` models, env-configurable) | ✅ wired; ⚠️ live calls need `GROQ_API_KEY` |
| **Local RAG corpus of real NASA NTRS papers** (102 documents, 16,168 page-anchored chunks, 5,444 pages, multi-tag investigations) | ✅ ingested & tested |
| **Hybrid retrieval** (TF-IDF + dense `all-MiniLM-L6-v2`, persisted embeddings, offline-capable) | ✅ ingested & benchmarked |
| Literature retrieval tool `search_scientific_literature` wired into NASA + document agents | ✅ tested |
| Retrieval evaluation harness (`scripts/evaluate.py`: hit@1=4/5, hit@3=5/5) | ✅ runs offline |
| Gold set + faithfulness harness (`data/gold/goldset.json`, 30 Q; `scripts/evaluate_faithfulness.py`) | ✅ retrieval/coverage metrics offline; live judging needs key |
| Gold-set results: hybrid source_hit@5=30/30, investigation_hit@5=30/30, keyword coverage 0.972 (TF-IDF: 29/30, 28/30, 0.972) | ✅ measured |
| NASA NTRS live retrieval (`search_nasa_research`, `retrieve_investigation_metadata`) | ✅ tested live |
| NASA Image & Video Library retrieval (`search_nasa_images`, no key; official FLEX/BASS-II/ACME imagery) | ✅ live-verified |
| Offline test suite | ✅ **85 passed, 2 skipped** (incl. parallel-execution perf test) |
| PSI (psi.nasa.gov) | ⚠️ portal only — no public JSON API; reported honestly |
| Document intelligence (PDF/DOCX/TXT/CSV/JSON) | ✅ tested |
| Deterministic statistics (Pandas/NumPy) | ✅ tested |
| ML suitability & metric evaluation | ✅ tested |
| Knowledge-graph extraction (graph-ready JSON) | ✅ tested |
| Multimodal figure analysis | ✅ implemented; returns `unavailable` without vision-capable model + key |
| Scientific reviewer (claims↔evidence, citation reachability, rejection) | ✅ tested |
| Sessions (ADK `InMemorySessionService`) | ✅ tested — in-memory only |
| Structured summarization (verified/unverified separation) | ✅ tested |
| Context/token budgets (tiktoken cl100k approximation) | ✅ tested |
| ADK developer UI (`adk web`) | ✅ app discovery verified |

## 2. The literature corpus

A real, local retrieval corpus built from **live-downloaded public NASA NTRS PDFs**:

- **102 unique documents** across **BASS/BASS-II (16), FLEX/FLEX-related (9), ACME (9), SAME/SAME-R (12), SAFFIRE (10), CIR (11), FLAME-SPREAD (9), FLAMMABILITY (8), COOL-FLAMES (12), SMOKE-DET (7), MAT-FLAM (10), FIRE-SAFETY (11)** — counts are investigation-tag memberships; documents fetched by several NTRS queries carry **all matching tags**, so filters stay honest
- **16,168 chunks** (~900 chars, page-anchored via pypdf), **5,444 PDF pages**
- **Hybrid retrieval**: TF-IDF (1–2 grams, sublinear) ⊕ dense MiniLM embeddings (`all-MiniLM-L6-v2`, 384-dim, CPU) with per-query min-max normalization, a 0.6/0.4 dense/sparse blend (`IGNIS_DENSE_WEIGHT`), and a small **title-overlap boost** (+0.15·overlap — BM25F-style field weighting; fixes title-literal queries like "Multi-User Droplet Combustion Apparatus"). Embeddings persist in `data/corpus/embeddings.npy` and auto-rebuild when chunks change. `IGNIS_DENSE=off` forces sparse-only; `auto` (default) falls back to TF-IDF **with a logged warning** — never silently
- Every passage carries `title`, `NTRS URL`, `page`, `investigations`, `retrieval` mode — cite as `[title | NTRS URL | p.N]`
- Dense leg is **included by default** and runs fully local/CPU; `uv sync` installs it (CPU-only torch via `uv.lock`, ~1.4 GB once). `IGNIS_DENSE=off` forces TF-IDF at runtime — no reinstall needed. Embeddings ship in the package, so no re-encoding is required.
- Benchmarks on the expanded corpus (5-query sanity / 30-question gold, k=5): **hit@1=4/5, hit@3=5/5 (hybrid and TF-IDF) · gold hybrid 30/30 source-hit, 30/30 investigation-hit, 0.972 coverage (TF-IDF 29/30, 28/30, 0.972)** — hybrid wins the harder metric on both corpora sizes
- Campaign queries are deliberately narrow (combustion/fire-safety) to avoid the aerosol-drift noise that a bare "smoke aerosol" search returned; a few off-target docs remain and are demoted by hybrid ranking, not deleted
- Known gaps, all reported in `data/corpus/ingest_report.json`: 2 newest PDFs fail pypdf parsing; 15 records have no public PDF (citation URLs listed) — none substituted or fabricated

```powershell
python scripts/ingest_corpus.py --refresh --max-per-investigation 12   # rebuild corpus (unions with curated IDs)
python scripts/evaluate.py                                            # 5-query benchmark
python scripts/evaluate_faithfulness.py                               # 30-question gold harness (offline)
python scripts/evaluate_faithfulness.py --live                        # + Groq answers (needs key)
```

PDFs are git-ignored (`data/corpus/`); provenance lives in `data/corpus/metadata.json` + `ingest_report.json`.

## 3. Project layout

```
ignis_adk/
├── README.md  requirements.txt  pytest.ini  .env.example  .env  main.py
├── ignis/
│   ├── agent.py                  # ADK app entry (`adk web` discovers 'ignis')
│   ├── agents/                   # root_orchestrator + 8 specialists
│   ├── tools/                    # 25 tools incl. search_scientific_literature, search_nasa_images
│   ├── corpus/                   # manifest.py (verified NTRS IDs), store.py (hybrid TF-IDF + dense index, multi-tag)
│   ├── schemas/  context/  orchestration/  sessions/  config/
├── scripts/ingest_corpus.py  scripts/evaluate.py  scripts/evaluate_faithfulness.py  scripts/run_agent.py
├── tests/                        # 85 passing offline tests (10 corpus/RAG tests included)
└── data/corpus/  data/gold/      # git-ignored PDFs/embeddings; gold set + harness docs
```

## 4. Setup (uv — the supported path)

```powershell
cd C:\path\to\ignis_adk
uv venv
uv sync                        # full install incl. CPU-only torch + sentence-transformers (~1.4 GB, one time)
copy .env.example .env         # PowerShell: copy  .env.example  .env   → then set GROQ_API_KEY=gsk_...
uv run pytest -q               # expect: 85 passed, 2 skipped (skipped = live Groq / live NTRS opt-ins)
uv run python main.py info     # config + corpus stats (never prints your key)
uv run adk web --host 127.0.0.1 --port 8000   # official ADK UI (localhost only)
```

Notes:
- **First dense query downloads the MiniLM model once** (~90 MB from Hugging Face), cached afterwards. Force TF-IDF meanwhile with `IGNIS_DENSE=off`.
- **pip fallback** (if uv is unavailable): `python -m venv .venv` → activate → `pip install -r requirements.txt` (CPU index line included) → `pip install -r` nothing else needed; then the same commands without the `uv run` prefix.
- **VS Code**: Ctrl+` opens the terminal → run the block above. **F5** in `launch.json` runs ask / info / a demo question / pytest debugging. Integrated pytest is preconfigured (`${workspaceFolder}/.venv`).
- Live Groq is the only thing that needs the key; the offline suite never touches the network.
- Corpus is pre-built and shipped (metadata + embeddings; no PDFs). To rebuild from NASA: `uv run python scripts/ingest_corpus.py --refresh --max-per-investigation 15` (network; ~10-15 min).

## 5. Run
## 5. Run

Prefix with `uv run` when using uv (e.g. `uv run python main.py ...`).

```powershell
python main.py ask "What did FLEX measure and what are the ISS fire safety implications?"
python main.py ask "Compare BASS-II and FLEX burning rates" --workflow B --json
python main.py repl
adk web --host 127.0.0.1 --port 8000     # official ADK UI, bind to localhost only
python scripts/run_agent.py "Find NASA research on SAME"
```

Tools added to agents: `search_scientific_literature(query, top_k, investigation)` (NASA +
document agents), `get_corpus_status()`, and `search_nasa_images(query, media_type, page_size)`
(NASA Image & Video Library — free, no API key; official ISS experiment imagery). Agents cite
passages inline as `[title | NTRS URL | p.N]`; when nothing matches, they must say so.

## 6. Performance & latency (honest)

**Zero latency is impossible** — every LLM turn pays network + inference time. IGNIS is engineered to be as fast as hosted inference allows, and no faster-than-physics claims are made:

- **Groq inference** (`openai/gpt-oss-120b` for general/reasoning work and `openai/gpt-oss-20b` for fast deterministic agents) — model IDs are configurable through `.env`.
- **Parallel specialist layers** (`WORKFLOW_LAYERS` in `routing.py`): independent specialists run concurrently via `asyncio.gather` with isolated branch sessions. Workflow B/D wall-time ≈ 2 layers × per-agent time instead of the sum of all agents. Verified by `test_parallel_layers_run_concurrently`.
- **ADK App + prefix caching** (`context_cache_config`, 8-turn TTL) exported in `ignis/agent.py` and used by the workflow runners — cuts repeat-prefix tokens when the provider supports caching; harmless no-op otherwise.
- **NTRS HTTP TTL cache** (300 s, `NASA_CACHE_TTL_SECONDS`) — repeated NASA searches skip the network.
- **Fast model for deterministic agents** — `data_analyst_agent` and `multimodal_agent` use `openai/gpt-oss-20b` by default.
- **Hybrid retrieval economics** — corpus embeddings are encoded once at ingest (~15–25 min on small CPUs for the 16k-chunk corpus, amortized to disk); query-time dense encoding adds only ~10–50 ms; sparse leg alone is sub-millisecond. On the expanded corpus: 5-query bench hit@1=4/5 for both legs; 30-question gold: hybrid 30/30 source-hit / 30/30 investigation-hit / 0.972 coverage vs TF-IDF 29/30 / 28/30 / 0.972 — hybrid strictly dominates.
- **Bounded tokens** — per-agent budgets, `MAX_OUTPUT_TOKENS=2048`, context truncation with source preservation.
- **`adk web` streams events** — perceived latency is partial, not all-or-nothing.
- What remains: one model turn per specialist minimum; reviewer adds a turn; Groq rate limits on free tier add waits. Measure yours with `python main.py ask ... --workflow B --json` (per-step timings are printed).

## 7. Honest limitations (updated)

1. **No Groq key at build time** — live multi-agent inference is wired but unexecuted here; offline E2E through the real ADK stack passes (FakeLlm + mocked HTTP). `--live` gold-harness mode also untested.
2. **Retrieval is now hybrid (TF-IDF + MiniLM) but still not entailment** — paraphrase ranking improves measurably (gold set), yet dense cosine can miss rare tokens; filters exist for precision; embedding is `all-MiniLM-L6-v2`, not a domain-tuned NASA model.
3. **Gold-set answers are drafts marked `expert_review_required`** — offline metrics (source/investigation hit, keyword coverage) prove the corpus can *support* answers; they do NOT prove any LLM answer is faithful or correct. That needs a judge model + expert-reviewed references.
4. **Two ACME fact sheets contribute 0 text chunks** (image-based), plus three 2024–26 PDFs fail pypdf parsing — they count as documents, not evidence.
5. **PSI still has no JSON API** — NTRS is the machine-readable source.
6. Token estimates approximate; sessions in-memory; corpus not a substitute for peer review.
7. Entity extraction rule-based; reviewer heuristics triage-grade; not spacecraft certification.
8. **Citation anchors prove provenance, not entailment.** Reviewer still required.
9. NTRS lexical search pulled in a few non-combustion aerosol/remote-sensing docs (e.g., laser-radar conference papers) — kept, not hidden; hybrid retrieval demotes them below combustion papers.
10. Duplicate NTRS records matched by several queries are stored **once, multi-tagged** — tag counts exceed unique document counts.

## 8. Tests

```powershell
pytest -q                              # 85 passed, 2 skipped
pytest -m integration -v               # live Groq (needs GROQ_API_KEY); set IGNIS_LIVE_NASA=1 for live NTRS
python scripts/evaluate.py             # corpus retrieval benchmark (5 queries)
python scripts/evaluate.py --tfidf     # sparse-only ablation
python scripts/evaluate_faithfulness.py          # 30-question gold harness (offline)
python scripts/evaluate_faithfulness.py --tfidf  # sparse-only ablation
python scripts/evaluate_faithfulness.py --live   # answers + heuristic proxy (needs GROQ_API_KEY)
```

## 8b. Delivery record (verified 2026-11-11)

| Item | Verified |
|---|---|
| Build | `uv sync` on a clean machine (CPU-only torch via `uv.lock` pin), pip fallback equally supported |
| Tests | **85 passed, 2 skipped** — skipped = live Groq + live NTRS, opt-in via `uv run pytest -m integration -v` |
| Corpus | 102 docs / 5,444 pages / 16,168 page-anchored chunks; PDFs excluded from the package by design (fully re-downloadable) |
| Gold harness (30 Q) | hybrid source_hit@5 = 30/30, investigation_hit@5 = 30/30, coverage 0.972; TF-IDF ablation reported beside it |
| Live path | Wired end-to-end but **never executed** with a real key — treat Groq answers as unverified until you run the integration tests |

**What cannot be claimed:** answers are not spacecraft certification; gold answers are `expert_review_required` drafts; faithfulness is evidenced, not certified; 2 newest PDFs fail pypdf parsing and 15 records have no public PDF (all listed in `data/corpus/ingest_report.json`); sessions are in-memory; MiniLM is not domain-tuned; citation anchors prove provenance, not entailment — reviewer remains required, triage-grade; Groq free-tier rate limits are your key's problem.

## 9. Security
- Keys from env only; never printed. `.env` git-ignored; corpus PDFs git-ignored.
- Documents are untrusted data; nothing in them is executed; tools validate inputs and bound retries.
- `adk web` has no auth — bind to 127.0.0.1 only.

## 10. What would make it even greater
An expert-reviewed gold reference set + LLM-judge faithfulness benchmark (the 30-question harness already
prints question/passage/answer triples for judging), metadata-aware reranking (investigation/fuel/pressure
filters), a domain-tuned dense encoder, durable sessions, graph-DB persistence (Neo4j), and live Groq E2E
benchmarks with real keys — in that order.
