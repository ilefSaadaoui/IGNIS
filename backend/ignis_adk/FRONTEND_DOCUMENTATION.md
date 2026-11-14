# IGNIS Frontend Integration Guide

This document defines the frontend behavior and integration contract for IGNIS -
Space Combustion Intelligence.

IGNIS is an evidence-grounded, multi-agent research system for NASA
microgravity-combustion and spacecraft fire-safety questions. The frontend must
present evidence, limitations, and review status as first-class information. It
must not present an AI draft as certified scientific or spacecraft-safety advice.

## 1. Backend integration status

The repository currently provides two supported ways to use the backend:

1. **Official ADK developer UI**
   ```powershell
   uv run adk web --host 127.0.0.1 --port 8000
   ```
   This is the working interactive UI for development. Bind it to localhost
   only because it has no application authentication layer.

2. **Programmatic workflow runner / CLI**
   ```powershell
   uv run python main.py ask "What did FLEX measure?" --workflow D --json
   ```
   `main.py --json` prints the `WorkflowResult.as_dict()` envelope described
   below. This is the current stable contract for a custom frontend adapter.

There is **currently no custom HTTP REST or WebSocket API** in this repository.
Do not implement frontend calls to undocumented routes such as
`/api/chat`, `/api/workflows`, or `/api/sessions`. If a browser-native
integration is required, add a thin authenticated backend adapter around
`ignis.orchestration.workflows.run_workflow`; keep that adapter separate from
the frontend and update this document with its real routes.

## 2. Recommended frontend architecture

Use four layers:

```text
UI components
  -> frontend state/store
    -> IGNIS adapter (HTTP when available, CLI bridge during development)
      -> run_workflow(...) / ADK
```

The UI should not know about Groq, ADK runners, specialist names, or prompt
construction. The adapter should:

- send a user question and optional workflow selection;
- preserve the returned `session_id`;
- normalize transport errors into an explicit error state;
- expose the complete result, including failed steps and limitations;
- never remove citations, evidence, uncertainty, or reviewer information.

For a browser deployment, the adapter should call a backend service rather than
spawning a local process. A CLI bridge is suitable for local development only.

## 3. Request contract

The current programmatic runner accepts:

| Field | Type | Required | Behavior |
|---|---|---:|---|
| `question` | `string` | yes | Natural-language research question |
| `workflow` | `"A"` through `"G"` or omitted | no | Omitted means deterministic auto-classification |
| `user_id` | `string` | no | Defaults to `local_user`; use an authenticated backend identity |
| `session_id` | `string` or omitted | no | Returned in the result; current sessions are in-memory |
| `include_reviewer` | `boolean` | no | Defaults to `true`; do not disable for scientific answers |

Frontend validation should reject an empty question and trim surrounding
whitespace. The frontend may show workflow selection as an advanced option;
auto-classification should be the default.

### Workflow choices

| Code | Name | Intended use |
|---|---|---|
| `A` | Research discovery | Find NASA publications, reports, or investigations |
| `B` | Experiment comparison | Compare investigations, documents, or measurements |
| `C` | ML suitability | Assess data readiness and machine-learning approaches |
| `D` | Scientific question | Answer a scientific question and assess fire-safety relevance |
| `E` | Knowledge graph | Extract entities and relationships from supplied text |
| `F` | Figure analysis | Analyze a supplied figure, chart, or image |
| `G` | Statistical analysis | Compute descriptive statistics or compare groups |

The backend's auto-classifier uses keyword matching and defaults to `D`.
Selecting a workflow explicitly is useful when a question contains ambiguous
terms.

## 4. Response contract

The CLI/programmatic workflow result has this shape:

```ts
type WorkflowStepStatus = "completed" | "failed" | "skipped";

interface WorkflowStepLog {
  step: number;
  agent: string;
  status: WorkflowStepStatus;
  output_excerpt: string;
  elapsed_seconds: number;
  error: string | null;
}

interface WorkflowResult {
  workflow: string;                 // "A" through "G"
  workflow_description: string;
  question: string;
  session_id: string;
  specialists_invoked: string[];
  final_answer: string;
  steps: WorkflowStepLog[];
  summary: Record<string, unknown>;
  limitations: string[];
}
```

The exact field names above are the serialized names from
`WorkflowResult.as_dict()`. Do not rename them in the adapter unless the
adapter exposes a separately documented frontend model.

### Important response semantics

- `final_answer` is text assembled from specialist outputs. It may contain
  inline citations such as `[title | NTRS URL | p.N]`.
- `steps` is the execution audit trail. A result can contain a usable answer
  and also have failed specialist steps.
- A failed step is not a successful empty result. Show the failure and its
  `error` to the user in an expandable diagnostic area.
- `limitations` are part of the scientific result, not incidental debug text.
  Keep them visible near the answer or behind an obvious "Limitations" control.
- `summary` contains the conversation summarizer's state and may include
  unverified claims. It is not a replacement for the answer or reviewer output.
- The CLI exits with status `2` if any workflow step fails. A frontend adapter
  should still parse and display the returned structured result when available,
  while marking the run as partially failed.

## 5. Agent and domain result models

Specialist outputs use Pydantic models in `ignis/schemas/`. When a future HTTP
adapter exposes structured specialist results, preserve these statuses:

```ts
type TaskStatus =
  | "completed"
  | "partial"
  | "unavailable"
  | "failed"
  | "requires_review";
```

Common agent results contain:

```ts
interface SourceRef {
  title?: string | null;
  url?: string | null;
  source?: string | null;       // e.g. "NASA NTRS", "document"
  identifier?: string | null;
  published?: string | null;
  accessed?: string | null;
}

interface AgentResult {
  agent_name: string;
  task_status: TaskStatus;
  summary: string;
  findings: string[];
  evidence: string[];
  sources: SourceRef[];
  limitations: string[];
  confidence_status: "high" | "moderate" | "low" | "unknown" | string;
  errors: string[];
}
```

Domain-specific screens should use the corresponding schema rather than
guessing fields:

- **Research discovery:** `DiscoveryResult`, `InvestigationRecord`,
  `InvestigationMetadata`
- **Documents and graphs:** `DocumentExtraction`, `ExperimentalConditions`,
  `ScientificFinding`, `GraphRepresentation`
- **Analysis and ML:** `DescriptiveStatistics`, `MissingValueReport`,
  `ComparisonResult`, `MLSuitabilityAssessment`, `ModelEvaluation`
- **Safety:** `MissionRelevanceAssessment`, `SafetyGapReport`
- **Figures:** `FigureAnalysis`
- **Review:** `ReviewReport`, `ClaimAssessment`, `CitationCheck`

For example, a research-results card should display
`InvestigationRecord.title`, `abstract`, `published`, `authors`, `source`, and
the `url` as a link. A graph visualization should use entity IDs for node and
edge references, not display names as implicit identifiers.

## 6. Scientific review and trust UI

The reviewer is normally the final workflow layer. The frontend should reserve
a dedicated review panel with these states:

| Reviewer state | Meaning | UI treatment |
|---|---|---|
| `approved` | Claims passed the configured review checks | Show a positive review badge, not a certification badge |
| `incomplete` | Review found missing context | Show missing context prominently |
| `rejected` | Output should not be relied on | Warn before presenting as an answer |
| `requires_review` | More evidence or expert review is needed | Show a strong review-required warning |

Display claim assessments individually:

- `supported`: show matched evidence and citation;
- `unsupported`: show the claim as unverified or omit it from a confident
  summary;
- `insufficient_evidence`: show the evidence gap;
- citation statuses such as `verified_reachable`, `unreachable`,
  `missing_url`, or `unchecked`.

The reviewer assesses evidence and citation reachability. It does not certify
spacecraft operations, medical safety, mission readiness, or regulatory
compliance.

## 7. Recommended screen states

### Before submission

- Question input with examples.
- Optional workflow selector, defaulting to `Auto`.
- Clear indication that answers are research assistance and require expert
  review.
- For document or figure analysis, show accepted input types and size limits
  supplied by the backend adapter.

### While running

Show progress from `steps` as it becomes available when the transport supports
streaming. Use the backend workflow description and specialist names rather
than inventing progress percentages. A safe progress display is:

```text
Running workflow D
1. NASA research: running
2. Document intelligence: queued
3. Fire safety: queued
4. Scientific review: queued
```

The backend executes agents in layers. Agents in the same layer may run in
parallel, so the UI must not assume that step numbers represent a strict
one-after-another timeline.

### After completion

Recommended order:

1. Answer summary
2. Evidence and citations
3. Scientific review
4. Limitations and uncertainty
5. Specialist execution details
6. Raw/advanced output

Do not hide a failed step, missing citation URL, unavailable vision model, or
review requirement behind a generic success toast.

### Error states

Use separate messages for:

- invalid question input;
- backend unavailable or request timeout;
- missing `GROQ_API_KEY` / provider authentication;
- a failed specialist step;
- no text output from a specialist;
- unavailable multimodal capability;
- partial result with limitations.

Never replace these with an empty answer or a fabricated fallback.

## 8. Citations, evidence, and links

The local corpus returns page-anchored evidence in the form:

```text
[title | NTRS URL | p.N]
```

The frontend should:

- render the citation as a clickable source when a valid URL exists;
- preserve the page number;
- provide a source drawer or details view with title, URL, source, identifier,
  publication date, and access date when available;
- distinguish retrieved evidence from AI interpretation;
- preserve the original citation text when parsing is uncertain.

A citation anchor proves provenance, not that the cited passage entails every
claim. Keep the reviewer result and limitations adjacent to citations.

## 9. Session behavior

Sessions are currently in-memory and are not durable across backend process
restarts. The frontend may store the returned `session_id` for the current
runtime, but must not promise history recovery after restart.

For a production frontend, the backend adapter must define:

- authenticated ownership of `user_id` and `session_id`;
- durable session storage;
- authorization checks before reading or continuing a session;
- expiration and deletion behavior;
- a documented streaming or polling protocol.

Do not put provider keys or backend environment variables in browser code.

## 10. Accessibility and interaction requirements

- Use semantic headings and landmarks for answer, evidence, review, and
  limitations sections.
- Make workflow and reviewer status available as text, not color alone.
- Provide keyboard-accessible citation links, accordions, graph controls, and
  image alternatives.
- Add an accessible live region for step-status changes during a run.
- Preserve long scientific text with readable line length and copyable content.
- Give figures meaningful alt text and expose the raw interpretation and
  uncertainty list.
- Do not auto-scroll away from a user's current reading position when a new
  specialist result arrives.

## 11. Local development checklist

From the repository root:

```powershell
uv sync
copy .env.example .env
# Set GROQ_API_KEY in .env for live inference.
uv run pytest -q
uv run python main.py info
uv run adk web --host 127.0.0.1 --port 8000
```

To inspect the JSON contract without a custom frontend:

```powershell
uv run python main.py ask `
  "What did FLEX measure and what are the ISS fire-safety implications?" `
  --workflow D --json
```

The offline test suite does not require live Groq inference. Live answers
require a real `GROQ_API_KEY`; the key value must never be logged or sent to
the browser.

## 12. Backend changes required before production browser integration

The current repository is ready for the ADK developer UI and local CLI
experimentation. Before connecting a deployed browser frontend, add and
document:

1. An authenticated HTTP or WebSocket adapter around `run_workflow`.
2. Request and response validation using the existing Pydantic schemas.
3. Explicit CORS and origin policy.
4. Rate limits, request timeouts, cancellation, and maximum input sizes.
5. Durable, authorized session storage.
6. A streaming event contract for step progress.
7. Structured error codes that preserve specialist failures and provider
   failures separately.
8. Server-side secret management and redacted logging.

Until those items exist, treat the CLI JSON output as a development adapter
contract, not a public API.

