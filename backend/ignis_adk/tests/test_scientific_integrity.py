"""Scientific-integrity guards: rejection paths, honesty of unavailable sources."""

from ignis.tools.review_tools import review_scientific_claims, validate_citations
from ignis.tools.nasa_tools import search_nasa_research
from ignis.context.summarizer import ConversationSummarizer


def test_reviewer_rejects_fabricated_findings():
    evidence = ["FLEX studied droplet combustion in microgravity aboard the ISS."]
    result = review_scientific_claims(
        ["The Moon's gravity eliminates all flame spread", "BASS used unicorn fuel"],
        evidence,
    )
    verdicts = {a["verdict"] for a in result["assessments"]}
    assert verdicts == {"unsupported"}
    assert result["verdict"] == "rejected"


def test_reviewer_approves_only_with_matching_evidence():
    evidence = ["BASS-II investigated burning rates of solid fuels in microgravity aboard the ISS."]
    result = review_scientific_claims(
        ["BASS-II investigated burning rates of solid fuels in microgravity"], evidence)
    assert result["verdict"] == "approved"


def test_missing_citation_url_not_marked_verified(mock_citation_head):
    r = validate_citations([{"title": "A claim with no source URL"}], check_reachability=True)
    assert r["checks"][0]["status"] == "missing_url"
    assert r["verdict"] == "incomplete"


def test_nasa_tools_never_fabricate_when_source_fails(monkeypatch):
    import ignis.tools.nasa_tools as nasa_tools

    class _Boom:
        def __call__(self, *a, **kw):
            raise nasa_tools.requests.ConnectionError("no route to host")

    monkeypatch.setattr(nasa_tools.requests, "get", _Boom())
    r = search_nasa_research("BASS", source="ntrs", limit=3)
    assert r["status"] in {"unavailable", "partial"}
    assert r["records"] == []
    assert r["errors"]


def test_summarizer_keeps_unverified_out_of_verified():
    s = ConversationSummarizer()
    s.set_question("Q")
    s.add_verified_finding("AI-generated speculation about Velcro flames", sources=[])
    assert s.summary.verified_findings == []
    assert any("Velcro flames" in c for c in s.summary.unverified_claims)
