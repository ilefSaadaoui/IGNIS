import sys, os

sys.path.insert(0, r"C:\Users\hp\Downloads\IGNIS_Final\frontend")
sys.path.insert(0, r"C:\Users\hp\Downloads\IGNIS_Final\backend\ignis_adk")
os.chdir(r"C:\Users\hp\Downloads\IGNIS_Final\backend\ignis_adk")

results = {}

# TASK 1 — Connecting backend APIs with Bilal's frontend
try:
    from ignis_client import backend_status
    ok, err = backend_status()
    results["TASK 1 - Backend APIs connected"] = ("✅ PASS", None) if ok else ("❌ FAIL", err)
except Exception as e:
    results["TASK 1 - Backend APIs connected"] = ("❌ FAIL", str(e))

# TASK 2 — Frontend-backend communication & data flow
try:
    from ignis.corpus.store import get_corpus_index
    ci = get_corpus_index()
    s = ci.stats()
    n_docs = s.get("documents", s.get("n_docs", "?"))
    n_chunks = s.get("chunks", s.get("n_chunks", "?"))
    has_dense = s.get("dense", {}).get("embeddings_ready", False)
    results["TASK 2 - Data flow (corpus)"] = ("✅ PASS", f"{n_docs} docs | {n_chunks} chunks | embeddings={has_dense}")
except Exception as e:
    results["TASK 2 - Data flow (corpus)"] = ("❌ FAIL", str(e))

# TASK 2b — Workflow routing
try:
    from ignis.orchestration.routing import workflow_for
    wf = workflow_for("What causes flame spread in microgravity?")
    results["TASK 2 - Workflow routing"] = ("✅ PASS", f"classified → workflow '{wf}'")
except Exception as e:
    results["TASK 2 - Workflow routing"] = ("❌ FAIL", str(e))

# TASK 3 — API integration testing
try:
    from ignis.tools.literature_tools import get_corpus_status
    status = get_corpus_status()
    n = status.get("corpus", {}).get("n_docs", status.get("n_docs", "?"))
    results["TASK 3 - Corpus API (get_corpus_status)"] = ("✅ PASS", f"{n} docs available")
except Exception as e:
    results["TASK 3 - Corpus API (get_corpus_status)"] = ("❌ FAIL", str(e))

# TASK 4 — All components work together (all 8 agents)
try:
    from ignis.agents import (
        nasa_research_agent, document_intelligence_agent,
        data_analyst_agent, ml_agent, knowledge_graph_agent,
        multimodal_agent, fire_safety_agent, scientific_reviewer_agent
    )
    results["TASK 4 - All 8 agents importable"] = ("✅ PASS", "nasa, document, data_analyst, ml, graph, multimodal, safety, reviewer")
except Exception as e:
    results["TASK 4 - All 8 agents importable"] = ("❌ FAIL", str(e))

# TASK 4b — All 10 frontend views
try:
    views = ["ask","data_ml","documents","figures","graph","home","research","reviewer","safety","system"]
    for v in views:
        __import__(f"views.{v}")
    results["TASK 4 - All 10 frontend views"] = ("✅ PASS", f"{len(views)}/10 views loaded")
except Exception as e:
    results["TASK 4 - All 10 frontend views"] = ("❌ FAIL", str(e))

# TASK 5 — AI integration (FastAPI server + ADK)
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("server", r"C:\Users\hp\Downloads\IGNIS_Final\backend\ignis_adk\server.py")
    if spec is not None and spec.loader is not None:
        srv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(srv)
        routes = [r.path for r in srv.app.routes]
        api_routes = [r for r in routes if r.startswith("/api")]
        results["TASK 5 - FastAPI server + AI endpoints"] = ("✅ PASS", f"title='{srv.app.title}' | {len(api_routes)} API routes: {api_routes}")
    else:
        results["TASK 5 - FastAPI server + AI endpoints"] = ("❌ FAIL", "Could not load server.py spec or loader")
except Exception as e:
    results["TASK 5 - FastAPI server + AI endpoints"] = ("❌ FAIL", str(e))

# Print results
print("\n" + "="*70)
print("  IGNIS — INTEGRATION AUDIT REPORT")
print("="*70)
for name, (status, detail) in results.items():
    print(f"\n{status}  {name}")
    if detail:
        print(f"        → {detail}")

passed = sum(1 for s, _ in results.values() if "PASS" in s)
print(f"\n{'='*70}")
print(f"  RESULT: {passed}/{len(results)} tasks passed")
print("="*70)
