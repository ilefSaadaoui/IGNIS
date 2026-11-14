from ignis.context.summarizer import ConversationSummarizer, ResearchSummary


def test_verified_finding_requires_source():
    s = ConversationSummarizer()
    s.set_question("Q")
    s.add_verified_finding("BASS measured flammability in microgravity", ["https://ntrs.nasa.gov/citations/20160000593"])
    s.add_verified_finding("Unsourced claim X", [])  # must demote, not enshrine
    assert len(s.summary.verified_findings) == 1
    assert any("Unsourced claim X" in c for c in s.summary.unverified_claims)
    assert "https://ntrs.nasa.gov/citations/20160000593" in s.summary.sources


def test_deduplication_and_version_bump():
    s = ConversationSummarizer()
    v0 = s.summary.version
    s.add_unverified_claim("claim A")
    s.add_unverified_claim("claim A")
    assert s.summary.unverified_claims.count("claim A") == 1
    assert s.summary.version > v0


def test_unresolved_questions_lifecycle():
    s = ConversationSummarizer()
    s.add_unresolved_question("What is the extinction limit?")
    s.resolve_unresolved_question("What is the extinction limit?")
    assert s.summary.unresolved_questions == []


def test_render_contains_sections():
    s = ConversationSummarizer()
    s.set_question("Compare FLEX and BASS", "microgravity combustion")
    s.add_verified_finding("Finding", ["https://ntrs.nasa.gov/x"])
    text = s.render()
    for marker in ["Question", "Verified findings", "Sources", "v"]:
        assert marker in text


def test_roundtrip_serialization():
    s = ConversationSummarizer()
    s.set_question("Q")
    s.add_verified_finding("F", ["https://ntrs.nasa.gov/y"])
    restored = ConversationSummarizer.from_dict(s.to_dict())
    assert restored.summary.current_question == "Q"
    assert restored.summary.verified_findings[0]["statement"] == "F"


def test_compress_history_bounds():
    s = ConversationSummarizer()
    events = [f"event {i}" for i in range(50)]
    digest = s.compress_history(events, max_items=6)
    assert "compressed" in digest
    assert digest.count("\n") <= 10
