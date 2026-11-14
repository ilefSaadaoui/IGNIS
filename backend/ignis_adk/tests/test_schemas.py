import pytest
from pydantic import ValidationError

from ignis.schemas import (
    AgentResult, TaskStatus, ReviewVerdict, ReviewReport, DiscoveryResult,
    normalize_investigation_name, KNOWN_INVESTIGATIONS,
)


def test_agent_result_defaults():
    r = AgentResult(agent_name="x", task_status=TaskStatus.COMPLETED)
    assert r.task_status == TaskStatus.COMPLETED
    assert r.findings == [] and r.sources == [] and r.errors == []


def test_task_status_values():
    assert {s.value for s in TaskStatus} == {"completed", "partial", "unavailable", "failed", "requires_review"}


def test_review_report_approval_logic():
    rep = ReviewReport(agent_name="scientific_reviewer_agent", task_status=TaskStatus.REQUIRES_REVIEW,
                       verdict=ReviewVerdict.REQUIRES_EVIDENCE)
    assert rep.approved is False
    rep2 = ReviewReport(agent_name="scientific_reviewer_agent", task_status=TaskStatus.COMPLETED,
                        verdict=ReviewVerdict.APPROVED)
    assert rep2.approved is True


def test_discovery_result_requires_status():
    with pytest.raises(ValidationError):
        DiscoveryResult(query="q", source="ntrs")  # missing task_status


def test_investigation_registry_complete_and_consistent():
    for name in ["BASS", "BASS-II", "FLEX", "ACME", "SAFFIRE", "SAME", "SAME-R"]:
        assert name in KNOWN_INVESTIGATIONS
        assert KNOWN_INVESTIGATIONS[name]["expansion"]
        assert KNOWN_INVESTIGATIONS[name]["verified"] is True


def test_normalize_investigation_aliases():
    assert normalize_investigation_name("SAFFIRE-III") == "SAFFIRE"
    assert normalize_investigation_name("same-r") == "SAME-R"
    assert normalize_investigation_name("BASS_II") == "BASS-II"
