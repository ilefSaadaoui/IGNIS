from ignis.context.token_manager import estimate_tokens, get_token_manager
from ignis.context.context_builder import ContextBuilder
from ignis.context.summarizer import ConversationSummarizer


def test_estimate_tokens_nonzero_and_scales():
    short = estimate_tokens("fire")
    long_ = estimate_tokens("fire " * 500)
    assert short > 0 and long_ > short * 50


def test_budget_reservation_and_oversize_detection():
    tm = get_token_manager()
    budget = tm.budget_for("ml_agent")
    assert budget.usable_input_tokens == budget.max_input_tokens
    big = tm.check("x " * 20000, "ml_agent")
    assert big.fits is False
    assert big.overage > 0


def test_truncation_preserves_head_and_tail():
    tm = get_token_manager()
    text = "HEAD" + ("filler " * 5000) + "TAIL"
    truncated, flag = tm.truncate_to_budget(text, "ml_agent")
    assert flag is True
    assert truncated.startswith("HEAD")
    assert "TAIL" in truncated
    assert "truncated" in truncated


def test_output_limit_enforced():
    tm = get_token_manager()
    text, flag = tm.enforce_output_limit("y" * 100000)
    assert flag is True and len(text) < 100000


def test_context_builder_dedupes_and_preserves_sources():
    summarizer = ConversationSummarizer()
    summarizer.set_question("Compare FLEX and BASS")
    builder = ContextBuilder()
    block = "Combustion of solids report https://ntrs.nasa.gov/citations/20160000593"
    result = builder.build_for_agent(
        "ml_agent",
        question="Compare FLEX and BASS",
        summarizer=summarizer,
        evidence_blocks=[block, block, "Another block"],
    )
    assert result.dropped_duplicates == 1
    assert "https://ntrs.nasa.gov/citations/20160000593" in result.preserved_sources


def test_context_builder_oversize_reports_limitation():
    builder = ContextBuilder()
    huge = "evidence block " + ("detail " * 30000)
    result = builder.build_for_agent("nasa_research_agent", question="q", evidence_blocks=[huge])
    assert result.truncated is True
    assert any("truncated" in lim.lower() for lim in result.limitations)
