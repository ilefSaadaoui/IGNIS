"""Tool-level tests — fully offline (NASA/citation HTTP is mocked)."""

from ignis.tools.data_tools import (
    analyze_missing_values, calculate_descriptive_statistics, compare_experimental_data, correlation_analysis,
)
from ignis.tools.document_tools import (
    analyze_document_text, extract_experimental_conditions, extract_research_findings, load_document,
)
from ignis.tools.graph_tools import build_graph_representation, extract_entities, extract_relationships
from ignis.tools.ml_tools import assess_ml_suitability, evaluate_model_metrics
from ignis.tools.nasa_tools import retrieve_investigation_metadata, search_nasa_research
from ignis.tools.source_tools import extract_research_source
from ignis.tools.review_tools import (
    identify_unsupported_findings, review_scientific_claims, validate_citations,
)
from ignis.tools.safety_tools import assess_mission_relevance, identify_safety_evidence_gaps

SAMPLE_TEXT = (
    "The FLEX experiment studied droplet extinction in microgravity aboard the International Space Station. "
    "Tests were conducted at 1 atm and 300 K with 21% O2 in nitrogen. PMMA samples burned at 25% oxygen. "
    "Results showed that extinction strain decreases with increasing oxygen concentration. "
    "We found that the flame extinguished at lower oxygen levels in microgravity than the 1-g baseline. "
    "See Figure 3 and Table 2. BASS-II studied PMMA flammability in microgravity."
)


# --- data tools --------------------------------------------------------------

def test_descriptive_statistics_basic():
    r = calculate_descriptive_statistics([1.0, 2.0, 3.0, 4.0, None], unit="kPa")
    assert r["status"] == "completed"
    assert r["count"] == 4 and r["missing"] == 1
    assert abs(r["mean"] - 2.5) < 1e-9 and r["unit"] == "kPa"


def test_descriptive_statistics_all_missing():
    r = calculate_descriptive_statistics([None, None])
    assert r["status"] == "partial" and r["mean"] is None


def test_descriptive_statistics_rejects_garbage():
    r = calculate_descriptive_statistics("not-a-list")
    assert r["status"] == "failed"


def test_missing_values_report():
    r = analyze_missing_values({"a": [1, None, 3], "b": [None, None, 2]})
    assert r["status"] == "completed"
    assert r["total_cells"] == 6 and r["total_missing"] == 3
    assert "recommendation" in r


def test_compare_groups():
    r = compare_experimental_data({"FLEX": [1.0, 2.0, 3.0], "BASS": [2.0, 3.0, 4.0]}, unit="mm/s")
    assert r["status"] == "completed"
    assert abs(r["mean_difference"] - 1.0) < 1e-9
    assert r["cohens_d"] is not None
    assert "FLEX" in r["interpretation"]


def test_compare_requires_two_groups():
    assert compare_experimental_data({"only": [1, 2]})["status"] == "failed"


def test_correlation():
    r = correlation_analysis({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    assert r["status"] == "completed"
    assert abs(r["pearson"]["x"]["y"] - 1.0) < 1e-9


# --- document tools ----------------------------------------------------------

def test_analyze_document_text_structure():
    r = analyze_document_text(SAMPLE_TEXT)
    assert r["status"] == "completed"
    assert "results" in r["sections_detected"] or "methods" in r["sections_detected"]
    assert any("fig" in f.lower() for f in r["figure_refs"])


def test_analyze_document_text_rejects_empty():
    assert analyze_document_text("   ")["status"] == "failed"


def test_extract_conditions_finds_pressure_temp_o2_gravity():
    r = extract_experimental_conditions(SAMPLE_TEXT)
    assert r["status"] == "completed"
    assert any("atm" in m for m in r["pressure"])
    assert any("O2" in m or "oxygen" in m.lower() or "%" in m for m in r["oxygen_concentration"])
    assert any("microgravity" in g.lower() or "space station" in g.lower() for g in r["gravity_environment"])


def test_extract_findings_with_evidence():
    r = extract_research_findings(SAMPLE_TEXT)
    assert r["status"] == "completed"
    assert r["finding_count"] >= 1
    for f in r["findings"]:
        assert f["evidence_sentence"] in SAMPLE_TEXT
        assert f["cue"]


def test_load_document_missing_file():
    r = load_document("/no/such/file.pdf")
    assert r["status"] == "failed"


def test_load_document_txt(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text(SAMPLE_TEXT, encoding="utf-8")
    r = load_document(str(p))
    assert r["status"] == "completed"
    assert "FLEX" in r["text"]


def test_load_document_unsupported(tmp_path):
    p = tmp_path / "note.exe"
    p.write_bytes(b"\x00\x01")
    r = load_document(str(p))
    assert r["status"] == "failed" and "unsupported" in r["errors"][0].lower()


# --- graph tools --------------------------------------------------------------

def test_graph_extraction_and_build():
    e = extract_entities(SAMPLE_TEXT)
    assert e["status"] == "completed"
    names = {x["name"] for x in e["entities"]}
    assert "FLEX" in names and "BASS-II" in names and "PMMA" in names
    rels = extract_relationships(SAMPLE_TEXT, e["entities"])
    assert rels["status"] == "completed"
    graph = build_graph_representation(e["entities"], rels["relationships"])
    assert graph["node_count"] == len(e["entities"])
    assert graph["edge_count"] == len(graph["edges"])
    for edge in graph["edges"]:
        node_ids = {n["id"] for n in graph["nodes"]}
        assert edge["source"] in node_ids and edge["target"] in node_ids


def test_graph_empty_input():
    assert extract_entities("  ")["status"] == "failed"


# --- ML tools -----------------------------------------------------------------

def test_ml_suitability_small_data_not_suitable():
    r = assess_ml_suitability(n_samples=20, n_features=3, target_type="regression")
    assert r["suitable"] is False
    assert r["limitations"]


def test_ml_suitability_leakage_blocks():
    r = assess_ml_suitability(n_samples=500, n_features=5, target_type="binary_classification",
                              has_leakage_indicator=True)
    assert r["suitable"] is False
    assert r["leakage_risks"]


def test_ml_suitability_ok():
    r = assess_ml_suitability(n_samples=800, n_features=8, target_type="regression",
                              missing_fraction=0.02)
    assert r["suitable"] is True
    assert r["recommended_approaches"]


def test_evaluate_metrics_regression():
    r = evaluate_model_metrics([1.0, 2.0, 3.0, 4.0], [1.1, 1.9, 3.2, 3.8], "regression")
    assert r["status"] == "completed"
    assert r["metrics"]["rmse"] > 0 and r["metrics"]["r2"] > 0.9


def test_evaluate_metrics_classification_and_mismatch():
    r = evaluate_model_metrics([0, 1, 1, 0], [0, 1, 0, 0], "classification")
    assert r["status"] == "completed"
    assert 0 <= r["metrics"]["accuracy"] <= 1
    assert evaluate_model_metrics([1, 2], [1], "regression")["status"] == "failed"


def test_evaluate_metrics_never_invents_values():
    # y_true/y_pred length mismatch must fail, never fabricate metrics
    r = evaluate_model_metrics([], [], "regression")
    assert r["status"] == "failed"


# --- NASA tools (mocked HTTP) ---------------------------------------------------

def test_search_nasa_research_offline(mock_ntrs_http):
    r = search_nasa_research("BASS microgravity combustion", source="ntrs", limit=2)
    assert r["status"] == "completed"
    assert r["retrieve_status"] == "ntrs_ok"
    assert r["total_matched"] == 2
    assert all(rec["url"].startswith("https://ntrs.nasa.gov/citations/") for rec in r["records"])
    assert r["records"][0]["source"] == "NASA NTRS"
    assert r["records"][0]["pdf_url"].endswith("20160000593.pdf")
    assert r["records"][0]["fulltext_url"].endswith("20160000593.txt")


def test_extract_research_source_plain_text(monkeypatch):
    import ignis.tools.source_tools as source_tools

    class Response:
        status_code = 200
        headers = {"Content-Type": "text/plain"}
        content = b"Results showed that flame spread decreased in microgravity."
        encoding = "utf-8"

    monkeypatch.setattr(source_tools.requests, "get", lambda *args, **kwargs: Response())
    result = extract_research_source("https://example.org/paper.txt", "test source")
    assert result["status"] == "completed"
    assert result["source"] == "test source"
    assert result["url"] == "https://example.org/paper.txt"
    assert result["text"].startswith("Results showed")


def test_search_nasa_psi_unavailable(mock_ntrs_http):
    r = search_nasa_research("combustion", source="psi")
    assert r["status"] == "unavailable"
    assert "PSI" in " ".join(r["limitations"])


def test_search_nasa_invalid_input(mock_ntrs_http):
    assert search_nasa_research("", source="ntrs")["status"] == "failed"
    assert search_nasa_research("x", source="bogus")["status"] == "failed"


def test_retrieve_investigation_known_and_unknown(mock_ntrs_http):
    r = retrieve_investigation_metadata("BASS-II", limit=2)
    assert r["expansion_verified"] is True
    assert "Burning and Suppression of Solids" in r["expansion"]
    assert len(r["records"]) == 2
    r2 = retrieve_investigation_metadata("MADE-UP-CAMPAIGN-99", limit=1)
    assert r2["expansion"] is None and r2["expansion_verified"] is False


# --- review tools -----------------------------------------------------------------

def test_validate_citations_mocked(mock_citation_head):
    r = validate_citations([
        {"url": "https://ntrs.nasa.gov/citations/20160000593", "title": "BASS-II results"},
        {"url": "https://broken.example.com/404", "title": "Missing"},
        {"title": "No URL citation"},
    ])
    by_status = {c["status"] for c in r["checks"]}
    assert "verified_reachable" in by_status and "unreachable" in by_status and "missing_url" in by_status
    assert r["verdict"] in {"incomplete", "partial", "requires_review"}


def test_review_claims_supported_vs_unsupported():
    evidence = ["BASS-II investigated burning rates of solid fuels in microgravity aboard the ISS."]
    r = review_scientific_claims(
        ["BASS-II investigated burning rates of solid fuels in microgravity",
         "Ignition temperatures on the Moon are always below 500 C"],
        evidence,
    )
    by_claim = {a["claim"]: a["verdict"] for a in r["assessments"]}
    assert by_claim["BASS-II investigated burning rates of solid fuels in microgravity"] == "supported"
    assert by_claim["Ignition temperatures on the Moon are always below 500 C"] == "unsupported"
    assert r["verdict"] in {"rejected", "requires_review"}


def test_identify_unsupported_findings():
    r = identify_unsupported_findings(["completely fabricated claim about unicorn flames"], ["BASS studied PMMA in microgravity."])
    assert r["unsupported_count"] == 1
    assert r["unsupported_findings"]


# --- safety tools ------------------------------------------------------------------

def test_mission_relevance_and_gaps():
    r = assess_mission_relevance(["PMMA flammability and smoke detection in microgravity"], "iss")
    assert r["status"] == "completed"
    assert 0.0 < r["relevance_score"] <= 1.0
    assert any("not a certified" in lim.lower() or "NOT a certified" in lim for lim in r["translation_limits"])
    gaps = identify_safety_evidence_gaps("SAME", ["smoke detector performance in microgravity"])
    assert gaps["status"] == "completed"
    assert gaps["missing_count"] >= 1
