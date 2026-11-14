from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

FRONTEND_DIR = Path(__file__).resolve().parent   # IGNIS_Final\frontend\
ROOT_DIR = FRONTEND_DIR.parent                   # IGNIS_Final\
CANDIDATE_PATHS = [
    FRONTEND_DIR,
    ROOT_DIR / "backend" / "ignis_adk",          # IGNIS_Final\backend\ignis_adk
    ROOT_DIR / "backend",
    ROOT_DIR / "ignis_adk_final" / "ignis_adk",  # legacy fallback
    ROOT_DIR,
]
for path in CANDIDATE_PATHS:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

st.set_page_config(
    page_title="IGNIS - Space Combustion Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

import components as ui
import ignis_client as api
import theme
from views import ask, data_ml, documents, figures, graph, home, research, reviewer, safety, system


# ── Warm-load corpus at startup so the first page render is instant ──────────
@st.cache_resource(show_spinner=False)
def _warm_corpus():
    """Pre-load the corpus index and TF-IDF matrix once per Streamlit process."""
    url = api.get_backend_url()
    if url:
        try:
            stats = api.corpus_stats()
            return {"documents": stats.get("documents", 102), "chunks": stats.get("chunks", 16168)}
        except Exception:
            return {"documents": 102, "chunks": 16168}
    try:
        from ignis.corpus.store import get_corpus_index
        idx = get_corpus_index()
        return {"documents": len(idx.documents), "chunks": len(idx.chunks)}
    except Exception:
        return {"documents": 0, "chunks": 0}

if not _warm_corpus.clear:  # first-run guard (cache_resource always returns)
    pass
_placeholder = st.empty()
if "corpus_ready" not in st.session_state:
    _placeholder.markdown(
        """
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    height:60vh;text-align:center;">
            <div style="width:48px;height:48px;border:4px solid #1f2a3a;border-top:4px solid #ff6b2c;
                        border-radius:50%;animation:spin 1s linear infinite;margin-bottom:1.5rem;"></div>
            <div style="color:#ff6b2c;font-size:1.4rem;font-weight:600;letter-spacing:0.05em;">
                Loading IGNIS corpus…
            </div>
            <div style="color:#94a3b8;font-size:0.9rem;margin-top:0.5rem;">
                102 documents · 16,168 chunks · TF-IDF + dense embeddings
            </div>
        </div>
        <style>@keyframes spin{to{transform:rotate(360deg)}}</style>
        """,
        unsafe_allow_html=True,
    )
    _warm_corpus()
    st.session_state["corpus_ready"] = True
    _placeholder.empty()
    st.rerun()
else:
    _warm_corpus()
    _placeholder.empty()
# ─────────────────────────────────────────────────────────────────────────────

NAV_ITEMS = [
    ("Home", home),
    ("Ask IGNIS", ask),
    ("Research", research),
    ("Documents", documents),
    ("Data & ML", data_ml),
    ("Knowledge Graph", graph),
    ("Figures", figures),
    ("Fire Safety", safety),
    ("Reviewer", reviewer),
    ("System", system),
]
NAV_LOOKUP = dict(NAV_ITEMS)


def render_setup(error: str | None) -> None:
    theme.page_header(
        "SETUP REQUIRED",
        "Connect the backend to launch IGNIS.",
        "The Streamlit shell is running, but the IGNIS backend could not be imported.",
    )
    ui.render_callout(
        "Backend import failed",
        [error or "Unknown import failure. Check the traceback in your terminal."],
        "bad",
    )
    st.markdown("### 1. Create the environment")
    st.code(
        "py -3.13 -m venv .venv\n"
        ".venv\\Scripts\\Activate.ps1\n"
        "pip install -r requirements.txt\n"
        "pip install -r frontend/requirements.txt",
        language="powershell",
    )
    st.markdown("### 2. Configure credentials")
    st.code(
        "# .env\nGROQ_API_KEY=your-real-key\nNASA_API_KEY=your-nasa-key",
        language="text",
    )
    st.markdown("### 3. Launch")
    st.code("streamlit run frontend/app.py", language="powershell")
    ui.render_callout(
        "Why this happens",
        [
            "The backend imports google-adk, pydantic, numpy, pandas and other dependencies.",
            "Python 3.13 is recommended; the default Python may not have compatible wheels.",
            "Run the commands from the repository root, not from the frontend folder.",
        ],
        "warn",
    )


theme.inject()

with st.sidebar:
    st.markdown(
        """
        <div class="ignis-sidebar-brand">
          <div class="ignis-sidebar-mark">I</div>
          <div>
            <div class="ignis-sidebar-name">IGNIS</div>
            <div class="ignis-sidebar-tag">Space Combustion Intelligence</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    nav = st.radio("Navigate", [label for label, _ in NAV_ITEMS], key="nav", label_visibility="collapsed")
    st.divider()
    ready, backend_error = api.backend_status()
    if ready:
        st.markdown(ui.badge("backend ready", "good"), unsafe_allow_html=True)
    else:
        st.markdown(ui.badge("backend unavailable", "bad"), unsafe_allow_html=True)
    history_count = len(st.session_state.get("run_history", []))
    st.caption(f"Session runs: {history_count}")
    st.caption("In-memory sessions only. No durability across runs.")
    st.caption("Research assistant - not a spacecraft certification tool.")

if not ready:
    render_setup(backend_error)
    st.stop()

view = NAV_LOOKUP.get(nav, home)
try:
    view.render()
except api.BackendUnavailable as exc:
    render_setup(str(exc))
except Exception as exc:
    st.error(f"Unexpected error: {exc.__class__.__name__}: {exc}")
    with st.expander("Traceback"):
        st.exception(exc)

st.divider()
st.caption(
    "IGNIS is a research assistant. It does not certify flight hardware or replace a fire-safety authority. "
    "Evidence support is not scientific truth. Sessions are in-memory only."
)
