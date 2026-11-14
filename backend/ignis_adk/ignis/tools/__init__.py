"""IGNIS functional tool registry. All tools are independently testable."""

from ignis.tools.nasa_tools import retrieve_investigation_metadata, search_nasa_images, search_nasa_research
from ignis.tools.literature_tools import get_corpus_status, search_scientific_literature
from ignis.tools.core_tools import search_core_research
from ignis.tools.document_tools import (
    analyze_document_text,
    extract_experimental_conditions,
    extract_research_findings,
    load_document,
)
from ignis.tools.source_tools import extract_research_source
from ignis.tools.data_tools import (
    analyze_missing_values,
    calculate_descriptive_statistics,
    compare_experimental_data,
    correlation_analysis,
)
from ignis.tools.ml_tools import assess_ml_suitability, evaluate_model_metrics
from ignis.tools.graph_tools import (
    build_graph_representation,
    extract_entities,
    extract_relationships,
)
from ignis.tools.multimodal_tools import analyze_scientific_figure, vision_model_supported
from ignis.tools.safety_tools import assess_mission_relevance, identify_safety_evidence_gaps
from ignis.tools.review_tools import (
    identify_unsupported_findings,
    review_scientific_claims,
    validate_citations,
)

__all__ = [name for name in dir() if not name.startswith("_")]
