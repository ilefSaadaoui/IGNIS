from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from ignis_client import (
    AGENT_LABELS,
    WORKFLOW_LABELS,
    extract_citations,
    extract_urls,
    extract_verdict,
    json_bytes,
    linkify_citations,
    split_final_answer,
)


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def badge(text: str, variant: str = "muted") -> str:
    return f'<span class="ignis-badge ignis-badge--{variant}">{esc(text)}</span>'


def status_variant(status: str | None) -> str:
    return {
        "completed": "good",
        "approved": "good",
        "failed": "bad",
        "rejected": "bad",
        "partial": "warn",
        "incomplete": "warn",
        "requires_review": "warn",
        "unavailable": "warn",
        "skipped": "muted",
    }.get((status or "").lower(), "muted")


def status_badge(status: str | None) -> str:
    label = (status or "unknown").replace("_", " ")
    return badge(label.upper(), status_variant(status))


def verdict_badge(verdict: str | None) -> str:
    if not verdict:
        return badge("NO VERDICT", "muted")
    return badge(verdict.replace("_", " ").upper(), status_variant(verdict))


def metric_card(label: str, value: Any, sub: str = "", accent: bool = False) -> str:
    value_class = "ignis-card__value ignis-card__value--ember" if accent else "ignis-card__value"
    return f"""
    <div class="ignis-card">
      <div class="ignis-card__label">{esc(label)}</div>
      <div class="{value_class}">{esc(value)}</div>
      <div class="ignis-card__sub">{esc(sub)}</div>
    </div>
    """


def render_metrics_row(items: list[dict[str, Any]]) -> None:
    if not items:
        return
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        with column:
            st.markdown(
                metric_card(
                    item.get("label", ""),
                    item.get("value", ""),
                    item.get("sub", ""),
                    bool(item.get("accent")),
                ),
                unsafe_allow_html=True,
            )


def render_callout(title: str, items: list[str], variant: str = "") -> None:
    if not items:
        return
    list_items = "".join(f"<li>{esc(item)}</li>" for item in items)
    st.markdown(
        f"""
        <div class="ignis-callout {('ignis-callout--' + variant) if variant else ''}">
          <div class="ignis-callout__title">{esc(title)}</div>
          <ul>{list_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_limitations(limitations: list[str]) -> None:
    render_callout("Limitations & uncertainty", limitations, "warn")


def render_errors(errors: list[str], title: str = "Errors") -> None:
    render_callout(title, errors, "bad")


def render_sources(citations: list[dict[str, str]], urls: list[str] | None = None) -> None:
    if not citations and not urls:
        st.markdown(
            '<div class="ignis-empty">No page-anchored citations were returned for this run. '
            "Treat the answer as unverified until sources are supplied.</div>",
            unsafe_allow_html=True,
        )
        return
    blocks = []
    for citation in citations:
        blocks.append(
            f"""
            <a class="ignis-source" href="{esc(citation['url'])}" target="_blank" rel="noopener noreferrer">
              <div class="ignis-source__title">{esc(citation['title'])}</div>
              <div class="ignis-source__meta">{esc(citation['page'])} · {esc(citation['url'])}</div>
            </a>
            """
        )
    for url in urls or []:
        if any(url == citation["url"] for citation in citations):
            continue
        blocks.append(
            f"""
            <a class="ignis-source" href="{esc(url)}" target="_blank" rel="noopener noreferrer">
              <div class="ignis-source__title">{esc(url)}</div>
              <div class="ignis-source__meta">Source link</div>
            </a>
            """
        )
    st.markdown("".join(blocks), unsafe_allow_html=True)


def render_steps(steps: list[dict[str, Any]]) -> None:
    if not steps:
        st.markdown('<div class="ignis-empty">No specialist steps were recorded.</div>', unsafe_allow_html=True)
        return
    blocks = []
    for step in steps:
        agent = step.get("agent", "unknown")
        label = AGENT_LABELS.get(agent, agent)
        status = step.get("status", "unknown")
        elapsed = step.get("elapsed_seconds", 0)
        index = step.get("step", "?")
        excerpt = (step.get("output_excerpt") or "").strip()
        error = (step.get("error") or "").strip()
        error_html = f'<div class="ignis-step__body" style="color:#fecdd3">Error: {esc(error)}</div>' if error else ""
        excerpt_html = (
            f"""
            <details style="margin-top:0.45rem">
              <summary style="font-size:0.74rem;color:#94a3b8;cursor:pointer">View output excerpt</summary>
              <div class="ignis-mono-block" style="margin-top:0.4rem">{esc(excerpt)}</div>
            </details>
            """
            if excerpt
            else ""
        )
        blocks.append(
            f"""
            <div class="ignis-step">
              <div class="ignis-step__head">
                <div class="ignis-step__agent">{esc(index)}. {esc(label)}</div>
                <div class="ignis-step__meta">
                  {status_badge(status)}
                  <span>{esc(elapsed)}s</span>
                </div>
              </div>
              {error_html}
              {excerpt_html}
            </div>
            """
        )
    st.markdown("".join(blocks), unsafe_allow_html=True)


def render_summary(summary: dict[str, Any]) -> None:
    if not summary:
        st.markdown('<div class="ignis-empty">No session summary was returned.</div>', unsafe_allow_html=True)
        return
    top = st.columns(3)
    with top[0]:
        st.metric("Summary version", summary.get("version", "—"))
    with top[1]:
        st.metric("Verified findings", len(summary.get("verified_findings", []) or []))
    with top[2]:
        st.metric("Unverified claims", len(summary.get("unverified_claims", []) or []))

    if summary.get("current_question"):
        st.markdown(f"**Question:** {summary['current_question']}")

    verified = summary.get("verified_findings") or []
    if verified:
        with st.expander(f"Verified findings ({len(verified)})", expanded=True):
            for finding in verified:
                statement = finding.get("statement", "")
                sources = ", ".join(finding.get("sources", []) or [])
                st.markdown(f"- {statement}  \n  <span class='ignis-muted'>Sources: {esc(sources)}</span>", unsafe_allow_html=True)

    unverified = summary.get("unverified_claims") or []
    if unverified:
        with st.expander(f"Unverified claims — do not present as fact ({len(unverified)})"):
            for claim in unverified:
                st.markdown(f"- {claim}")

    actions = summary.get("previous_agent_actions") or []
    if actions:
        with st.expander(f"Agent action log ({len(actions)})"):
            frame = pd.DataFrame(actions)
            st.dataframe(frame, use_container_width=True, hide_index=True)

    sources = summary.get("sources") or []
    if sources:
        with st.expander(f"Accumulated sources ({len(sources)})"):
            for source in sources:
                st.markdown(f"- {source}")

    limitations = summary.get("limitations") or []
    if limitations:
        render_limitations(limitations)


def _sections_for_result(result: dict[str, Any]) -> dict[str, Any]:
    split = split_final_answer(result.get("final_answer", ""), result.get("question", ""))
    steps = result.get("steps", []) or []
    reviewer_step = next(
        (step for step in steps if step.get("agent") == "scientific_reviewer_agent"),
        None,
    )
    review_text = split["sections"].get("scientific_reviewer_agent", "")
    if not review_text and reviewer_step:
        review_text = reviewer_step.get("output_excerpt", "")
    verdict = extract_verdict(review_text) or extract_verdict(result.get("final_answer", ""))
    return {
        "body": split["body"],
        "sections": split["sections"],
        "review_text": review_text,
        "verdict": verdict,
        "steps": steps,
    }


def render_result(result: dict[str, Any], key_prefix: str = "result") -> None:
    if result.get("_error") or not result.get("final_answer"):
        render_errors(result.get("errors") or ["No answer was produced."], "Workflow failed")
        steps = result.get("steps", []) or []
        if steps:
            with st.expander("Execution audit", expanded=True):
                render_steps(steps)
        return

    parts = _sections_for_result(result)
    steps = parts["steps"]
    workflow_code = result.get("workflow", "?")
    workflow_name = WORKFLOW_LABELS.get(workflow_code, result.get("workflow_description", ""))
    session_id = result.get("session_id", "-")
    total_seconds = round(sum(float(step.get("elapsed_seconds") or 0) for step in steps), 2)
    overall = result.get("status") or (
        "partial" if any(step.get("status") == "failed" for step in steps) else "completed"
    )

    workflow_badge = badge(f"Workflow {workflow_code} · {workflow_name}", "ember")
    steps_badge = badge(f"{len(steps)} specialist steps", "info")
    time_badge = badge(f"{total_seconds}s total", "muted")
    session_badge = badge(f"session {session_id}", "muted")
    st.markdown(
        f"""
        <div class="ignis-metric-strip">
          {workflow_badge}
          {status_badge(overall)}
          {steps_badge}
          {time_badge}
          {session_badge}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"### {result.get('question', 'Research question')}")

    if parts["verdict"]:
        variant = status_variant(parts["verdict"])
        titles = {
            "approved": "Reviewer verdict: APPROVED — evidence-backed, not a certification.",
            "incomplete": "Reviewer verdict: INCOMPLETE — missing context must be resolved.",
            "rejected": "Reviewer verdict: REJECTED — do not rely on this output.",
            "requires_review": "Reviewer verdict: REQUIRES REVIEW — evidence is partial.",
        }
        messages = {
            "approved": "Claims passed the configured review checks. Approval is evidence support, not scientific certification.",
            "incomplete": "The reviewer found missing context. Treat the answer as provisional until it is resolved.",
            "rejected": "The reviewer rejected this output. Do not present it as a reliable scientific answer.",
            "requires_review": "More evidence or expert review is needed before relying on this answer.",
        }
        render_callout(
            titles.get(parts["verdict"], f"Reviewer verdict: {parts['verdict']}"),
            [messages.get(parts["verdict"], "Review the claim assessments before use.")],
            variant,
        )
        st.markdown(verdict_badge(parts["verdict"]), unsafe_allow_html=True)
    elif result.get("_meta", {}).get("include_reviewer") is False:
        render_callout("Reviewer skipped", ["This run was executed without the scientific reviewer step."], "warn")

    with st.container(border=True):
        st.caption("ANSWER · evidence-grounded draft — expert review required")
        st.markdown(linkify_citations(parts["body"]))

    limitations = result.get("limitations") or []
    render_limitations(limitations)
    if result.get("errors"):
        render_errors(result["errors"])

    body_and_sections = parts["body"] + "\n" + "\n".join(parts["sections"].values())
    citations = extract_citations(body_and_sections)
    urls = [url for url in extract_urls(body_and_sections) if not any(url == c["url"] for c in citations)]

    tabs = st.tabs(
        ["Evidence & sources", "Scientific review", "Execution audit", "Session summary", "Raw / advanced"]
    )

    with tabs[0]:
        st.markdown(f"**{len(citations)} page-anchored citation(s) detected**")
        render_sources(citations, urls)

    with tabs[1]:
        if parts["review_text"]:
            st.markdown(verdict_badge(parts["verdict"]), unsafe_allow_html=True)
            st.markdown(linkify_citations(parts["review_text"]))
        else:
            st.markdown(
                '<div class="ignis-empty">No reviewer output was produced for this run.</div>',
                unsafe_allow_html=True,
            )

    with tabs[2]:
        render_steps(steps)
        other_sections = {
            key: value
            for key, value in parts["sections"].items()
            if key != "scientific_reviewer_agent" and value.strip()
        }
        if other_sections:
            st.markdown("#### Specialist outputs")
            for agent, output in other_sections.items():
                with st.expander(AGENT_LABELS.get(agent, agent)):
                    st.markdown(linkify_citations(output))

    with tabs[3]:
        render_summary(result.get("summary") or {})

    with tabs[4]:
        st.download_button(
            "Download result JSON",
            data=json_bytes(result),
            file_name=f"ignis_{workflow_code}_{session_id}.json",
            mime="application/json",
            key=f"{key_prefix}_download",
        )
        st.json(result)


def render_records(records: list[dict[str, Any]]) -> None:
    if not records:
        st.markdown('<div class="ignis-empty">No records returned.</div>', unsafe_allow_html=True)
        return
    blocks = []
    for record in records:
        title = record.get("title", "(untitled)")
        url = record.get("url") or record.get("pdf_url") or record.get("fulltext_url")
        title_html = f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(title)}</a>' if url else esc(title)
        meta_bits = []
        if record.get("identifier"):
            meta_bits.append(badge(f"ID {record['identifier']}", "muted"))
        if record.get("source"):
            meta_bits.append(badge(record["source"], "info"))
        if record.get("record_type"):
            meta_bits.append(badge(record["record_type"], "muted"))
        if record.get("published"):
            meta_bits.append(badge(f"Published {record['published'][:10]}", "muted"))
        if record.get("center"):
            meta_bits.append(badge(record["center"], "muted"))
        if record.get("distribution"):
            meta_bits.append(badge(record["distribution"], "warn" if record["distribution"] != "PUBLIC" else "good"))
        authors = record.get("authors") or []
        if authors:
            meta_bits.append(badge(", ".join(authors[:3]) + (" +" if len(authors) > 3 else ""), "muted"))
        keywords = record.get("keywords") or []
        keyword_html = "".join(badge(keyword, "muted") for keyword in keywords[:8])
        abstract = record.get("abstract") or ""
        abstract_html = f'<div class="ignis-record__abstract">{esc(abstract)}</div>' if abstract else ""
        links = []
        if record.get("url"):
            links.append(f'<a href="{esc(record["url"])}" target="_blank" rel="noopener noreferrer">NTRS record</a>')
        if record.get("pdf_url"):
            links.append(f'<a href="{esc(record["pdf_url"])}" target="_blank" rel="noopener noreferrer">PDF</a>')
        if record.get("fulltext_url"):
            links.append(f'<a href="{esc(record["fulltext_url"])}" target="_blank" rel="noopener noreferrer">Full text</a>')
        links_html = f'<div class="ignis-source__meta">{" · ".join(links)}</div>' if links else ""
        blocks.append(
            f"""
            <div class="ignis-record">
              <div class="ignis-record__title">{title_html}</div>
              <div class="ignis-record__meta">{"".join(meta_bits)}</div>
              {links_html}
              {abstract_html}
              <div class="ignis-record__meta">{keyword_html}</div>
            </div>
            """
        )
    st.markdown("".join(blocks), unsafe_allow_html=True)


def render_passages(passages: list[dict[str, Any]]) -> None:
    if not passages:
        st.markdown('<div class="ignis-empty">No corpus passages matched.</div>', unsafe_allow_html=True)
        return
    blocks = []
    for passage in passages:
        investigations = passage.get("investigations") or [passage.get("investigation", "")]
        tags = "".join(badge(tag, "ember") for tag in investigations if tag)
        blocks.append(
            f"""
            <div class="ignis-passage">
              <div class="ignis-passage__head">
                <span>{esc(passage.get('title', ''))}</span>
                <span>{esc(passage.get('page') or '—')} · score {esc(passage.get('score', '—'))} · {esc(passage.get('retrieval', '—'))}</span>
              </div>
              <div>{tags}</div>
              <div style="margin-top:0.45rem">{esc(passage.get('text', ''))}</div>
              <div class="ignis-source__meta"><a href="{esc(passage.get('url', ''))}" target="_blank" rel="noopener noreferrer">{esc(passage.get('url', ''))}</a></div>
            </div>
            """
        )
    st.markdown("".join(blocks), unsafe_allow_html=True)


def render_media_items(items: list[dict[str, Any]]) -> None:
    if not items:
        st.markdown('<div class="ignis-empty">No NASA media returned.</div>', unsafe_allow_html=True)
        return
    for index in range(0, len(items), 2):
        columns = st.columns(2)
        for column, item in zip(columns, items[index : index + 2]):
            with column:
                preview = item.get("preview_url")
                if preview:
                    try:
                        st.image(preview, use_container_width=True)
                    except Exception:
                        st.markdown('<div class="ignis-empty">Preview unavailable.</div>', unsafe_allow_html=True)
                st.markdown(f"**{item.get('title', '(untitled)')}**")
                meta_bits = []
                if item.get("date_created"):
                    meta_bits.append(badge(item["date_created"][:10], "muted"))
                if item.get("center"):
                    meta_bits.append(badge(item["center"], "muted"))
                if meta_bits:
                    st.markdown("".join(meta_bits), unsafe_allow_html=True)
                if item.get("description"):
                    with st.expander("Description"):
                        st.write(item["description"])
                if item.get("collection_url"):
                    st.markdown(f"[Open NASA media record]({item['collection_url']})")


def render_conditions(conditions: dict[str, Any]) -> None:
    if not conditions or conditions.get("status") == "failed":
        render_errors((conditions or {}).get("errors", ["Condition extraction failed."]))
        return
    categories = [
        ("pressure", "Pressure"),
        ("temperature", "Temperature"),
        ("oxygen_concentration", "Oxygen"),
        ("gravity_environment", "Gravity"),
        ("fuel_or_material", "Fuel / material"),
        ("flow_velocity", "Flow velocity"),
        ("atmosphere", "Atmosphere"),
    ]
    rows = []
    for key, label in categories:
        values = conditions.get(key) or []
        rows.append(
            {
                "category": label,
                "matches": ", ".join(values) if values else "—",
                "count": len(values),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    raw = conditions.get("raw_matches") or []
    if raw:
        with st.expander(f"Matched spans with context ({len(raw)})"):
            st.dataframe(pd.DataFrame(raw), use_container_width=True, hide_index=True)
    missing = conditions.get("missing_categories") or []
    if missing:
        render_callout("Not found in this document", [item.replace("_", " ") for item in missing], "warn")


def render_findings(findings: dict[str, Any]) -> None:
    if not findings or findings.get("status") == "failed":
        render_errors((findings or {}).get("errors", ["Finding extraction failed."]))
        return
    items = findings.get("findings") or []
    if not items:
        st.markdown('<div class="ignis-empty">No cue-phrase findings matched.</div>', unsafe_allow_html=True)
    for index, finding in enumerate(items, start=1):
        with st.expander(f"{index}. {finding.get('statement', '')[:120]}"):
            st.markdown(f"**Cue:** `{finding.get('cue', '')}`")
            st.markdown(finding.get("evidence_sentence", ""))
    render_limitations(findings.get("limitations") or [])


def _dot_quote(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def render_graph(payload: dict[str, Any]) -> None:
    graph = (payload or {}).get("graph") or {}
    entities = (payload or {}).get("entities") or {}
    relationships = (payload or {}).get("relationships") or {}
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    if not nodes:
        st.markdown('<div class="ignis-empty">No entities were extracted from this text.</div>', unsafe_allow_html=True)
        render_limitations((graph.get("limitations") or []) + (entities.get("limitations") or []))
        return

    palette = {
        "INVESTIGATION": "#ff6b2c",
        "MATERIAL": "#60a5fa",
        "FACILITY": "#34d399",
        "INSTRUMENT": "#a78bfa",
        "PHENOMENON": "#ffd166",
        "CONDITION": "#fb7185",
        "ORGANIZATION": "#f472b6",
        "MEASUREMENT": "#22d3ee",
        "UNKNOWN": "#94a3b8",
    }
    nodes = nodes[:80]
    visible = {node["id"] for node in nodes}
    edges = [edge for edge in edges if edge.get("source") in visible and edge.get("target") in visible][:140]

    lines = [
        "digraph IGNIS {",
        '  bgcolor="transparent";',
        '  rankdir="LR";',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#1f2a3a"];',
        '  edge [color="#475569", fontname="Helvetica", fontsize=8];',
    ]
    for node in nodes:
        color = palette.get(node.get("type", "UNKNOWN"), "#94a3b8")
        label = _dot_quote(f"{node.get('name', '')}\n({node.get('type', '')})")
        lines.append(f'  "{_dot_quote(node["id"])}" [label="{label}", fillcolor="{color}", fontcolor="#0b1018"];')
    for edge in edges:
        lines.append(
            f'  "{_dot_quote(edge["source"])}" -> "{_dot_quote(edge["target"])}" [label="{_dot_quote(edge.get("type", ""))}"];'
        )
    lines.append("}")
    dot = "\n".join(lines)

    node_count = graph.get("node_count", len(nodes))
    edge_count = graph.get("edge_count", len(edges))
    strip = "".join(badge(f"{key}: {value}", "muted") for key, value in (graph.get("node_types") or {}).items())
    node_badge = badge(f"{node_count} nodes", "ember")
    edge_badge = badge(f"{edge_count} edges", "info")
    st.markdown(
        f'<div class="ignis-metric-strip">{node_badge}{edge_badge}{strip}</div>',
        unsafe_allow_html=True,
    )
    st.graphviz_chart(dot, use_container_width=True)
    st.markdown(
        f'<div class="ignis-graph-note">Rendering {len(nodes)} of {node_count} nodes. '
        "Rule-based extraction — low confidence, review required.</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Nodes"):
        st.dataframe(pd.DataFrame(nodes), use_container_width=True, hide_index=True)
    with st.expander("Edges"):
        st.dataframe(pd.DataFrame(edges), use_container_width=True, hide_index=True)
    render_limitations((graph.get("limitations") or []) + (relationships.get("limitations") or []))


def render_safety_gaps(payload: dict[str, Any]) -> None:
    if not payload or payload.get("status") == "failed":
        render_errors((payload or {}).get("errors", ["Safety gap analysis failed."]))
        return
    gaps = payload.get("gaps") or []
    blocks = []
    for gap in gaps:
        blocks.append(
            f"""
            <div class="ignis-step">
              <div class="ignis-step__head">
                <div class="ignis-step__agent">{esc(gap.get('label') or gap.get('dimension'))}</div>
                {status_badge(gap.get('status'))}
              </div>
              <div class="ignis-step__body">{esc(gap.get('note', ''))}</div>
            </div>
            """
        )
    if blocks:
        st.markdown("".join(blocks), unsafe_allow_html=True)
    st.markdown(
        f"**Missing:** {payload.get('missing_count', 0)} · **Partial:** {payload.get('partial_count', 0)}"
    )
    if payload.get("summary"):
        st.info(payload["summary"])
    render_limitations(payload.get("limitations") or [])


def render_claim_review(payload: dict[str, Any]) -> None:
    if not payload or payload.get("status") == "failed":
        render_errors((payload or {}).get("errors", ["Claim review failed."]))
        return
    verdict_html = verdict_badge(payload.get("verdict"))
    unsupported_html = badge(f"unsupported: {payload.get('unsupported_count', 0)}", "warn")
    st.markdown(f"{verdict_html} {unsupported_html}", unsafe_allow_html=True)
    for assessment in payload.get("assessments") or []:
        verdict = assessment.get("verdict", "")
        with st.expander(f"{verdict.replace('_', ' ').upper()} · {assessment.get('claim', '')[:110]}"):
            st.markdown(assessment.get("claim", ""))
            if assessment.get("reason"):
                st.caption(assessment["reason"])
            for evidence in assessment.get("matched_evidence") or []:
                st.markdown(f"> {evidence}")
    render_limitations(payload.get("limitations") or [])


def render_citation_checks(payload: dict[str, Any]) -> None:
    if not payload or payload.get("status") == "failed":
        render_errors((payload or {}).get("errors", ["Citation validation failed."]))
        return
    verdict_html = verdict_badge(payload.get("verdict"))
    reachable_html = badge(f"{payload.get('reachable', 0)}/{payload.get('total', 0)} reachable", "info")
    st.markdown(f"{verdict_html} {reachable_html}", unsafe_allow_html=True)
    checks = payload.get("checks") or []
    if checks:
        rows = [
            {
                "citation": check.get("citation", ""),
                "status": check.get("status", ""),
                "detail": check.get("detail", ""),
                "url": check.get("url", ""),
            }
            for check in checks
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    render_limitations(payload.get("limitations") or [])


def render_ml_suitability(payload: dict[str, Any]) -> None:
    if not payload or payload.get("status") == "failed":
        render_errors((payload or {}).get("errors", ["ML assessment failed."]))
        return
    suitable = bool(payload.get("suitable"))
    render_callout(
        "ML suitability",
        [payload.get("summary", "")],
        "good" if suitable else "warn",
    )
    if payload.get("recommended_approaches"):
        st.markdown("**Recommended approaches**")
        for item in payload["recommended_approaches"]:
            st.markdown(f"- {item}")
    if payload.get("target_candidates"):
        st.markdown("**Target candidates**")
        for item in payload["target_candidates"]:
            st.markdown(f"- {item}")
    columns = st.columns(2)
    with columns[0]:
        if payload.get("feature_engineering"):
            with st.expander("Feature engineering", expanded=True):
                for item in payload["feature_engineering"]:
                    st.markdown(f"- {item}")
        if payload.get("leakage_risks"):
            with st.expander("Leakage risks", expanded=True):
                for item in payload["leakage_risks"]:
                    st.markdown(f"- {item}")
    with columns[1]:
        if payload.get("data_requirements"):
            with st.expander("Data requirements", expanded=True):
                for item in payload["data_requirements"]:
                    st.markdown(f"- {item}")
        if payload.get("limitations"):
            with st.expander("Limitations", expanded=True):
                for item in payload["limitations"]:
                    st.markdown(f"- {item}")


def render_model_metrics(payload: dict[str, Any]) -> None:
    if not payload or payload.get("status") == "failed":
        render_errors((payload or {}).get("errors", ["Metric evaluation failed."]))
        return
    metrics = payload.get("metrics") or {}
    if payload.get("summary"):
        st.info(payload["summary"])
    if metrics:
        keys = list(metrics.keys())
        for index in range(0, len(keys), 4):
            columns = st.columns(min(4, len(keys) - index))
            for column, key in zip(columns, keys[index : index + 4]):
                with column:
                    st.metric(key.upper(), round(float(metrics[key]), 4) if isinstance(metrics[key], (int, float)) else metrics[key])
    render_limitations((payload.get("uncertainty") or []) + (payload.get("overfitting_indicators") or []) + (payload.get("limitations") or []))


def render_json_block(payload: Any, filename: str, label: str = "Raw JSON") -> None:
    st.download_button(
        "Download JSON",
        data=json_bytes(payload),
        file_name=filename,
        mime="application/json",
        key=f"download_{filename}",
    )
    with st.expander(label):
        st.json(payload)


def render_tool_status(payload: dict[str, Any], label: str) -> bool:
    if not payload:
        return False
    status = payload.get("status", "completed")
    if status == "failed":
        render_errors(payload.get("errors") or [f"{label} failed."], f"{label} failed")
        return False
    if payload.get("errors"):
        render_errors(payload["errors"], f"{label} errors")
    if payload.get("limitations"):
        render_limitations(payload["limitations"])
    return True


def render_corpus_banner(stats: dict[str, Any]) -> None:
    if not stats:
        return
    dense = stats.get("dense") or {}
    dense_label = "hybrid" if dense.get("embeddings_ready") else "TF-IDF"
    render_metrics_row(
        [
            {
                "label": "Corpus documents",
                "value": stats.get("documents", 0),
                "sub": "Real NASA NTRS papers",
                "accent": True,
            },
            {
                "label": "Page-anchored chunks",
                "value": f"{stats.get('chunks', 0):,}",
                "sub": "Searchable evidence passages",
            },
            {
                "label": "Retrieval mode",
                "value": dense_label,
                "sub": dense.get("model", "TF-IDF"),
            },
        ]
    )
