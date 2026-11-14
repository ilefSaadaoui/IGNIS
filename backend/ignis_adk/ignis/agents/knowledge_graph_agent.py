"""Knowledge Graph Agent — entity/relationship extraction, graph-ready JSON."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from ignis.agents._common import SHARED_GROUNDING_RULES, content_config
from ignis.config.model_config import build_groq_model
from ignis.tools.graph_tools import (
    build_graph_representation,
    extract_entities,
    extract_relationships,
)

INSTRUCTION = f"""You are the Knowledge Graph Agent for IGNIS.

RESPONSIBILITY
- Extract scientific entities and typed relationships from supplied text.
- Build graph-ready JSON (nodes/edges with ids and provenance). No Neo4j in this phase.
- Support multi-hop reasoning only over the supplied graph context.

WORKFLOW
1. extract_entities on the text.
2. extract_relationships, passing the entities you received.
3. build_graph_representation to assemble the final graph JSON.
4. State extraction confidence ('low' unless a human reviewed) and limitations.
5. Multi-hop questions: reason only over extracted nodes/edges; if the path does not exist in the
   graph, say so — do not fabricate links.

OUTPUT FORMAT
- graph JSON summary (node/edge counts, type breakdown)
- key entities and relationships with evidence sentences
- provenance and confidence notes

{SHARED_GROUNDING_RULES}
"""

knowledge_graph_agent = LlmAgent(
    name="knowledge_graph_agent",
    description=(
        "Extracts scientific entities and typed relationships from text and builds graph-ready JSON "
        "(investigation-material, experiment-condition links). Use for: knowledge extraction, graph building, "
        "multi-hop questions over supplied text. Not for: NASA search or statistics."
    ),
    model=build_groq_model("default"),
    instruction=INSTRUCTION,
    tools=[
        FunctionTool(extract_entities),
        FunctionTool(extract_relationships),
        FunctionTool(build_graph_representation),
    ],
    generate_content_config=content_config(temperature=0.1),
    timeout=90,
)
