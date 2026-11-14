from __future__ import annotations

import streamlit as st

import components as ui
import data
import ignis_client as api
import theme

EXAMPLES = [
    (
        "Flame spread",
        "How does flame spread over thin solid fuels in microgravity, and which NASA experiments support it?",
    ),
    (
        "Safety gaps",
        "What are the main fire-safety evidence gaps for crewed spacecraft, and which investigations address them?",
    ),
    (
        "Compare missions",
        "Compare the BASS and FLEX microgravity flame spread investigations and cite the evidence.",
    ),
    (
        "Challenge a claim",
        "Is it established that microgravity always increases flame spread rate? Review the evidence.",
    ),
]


def _set_question(question: str) -> None:
    st.session_state["ask_question"] = question


def render() -> None:
    theme.page_header(
        "ASK IGNIS",
        "Route a question to the right specialists.",
        "Auto-classification selects one of seven workflows. Every answer is evidence-grounded, instrumented and reviewed.",
    )

    catalog = data.workflow_catalog()
    options = ["auto"] + [item["code"] for item in catalog]
    labels = {"auto": "Auto - classify this question"}
    for item in catalog:
        labels[item["code"]] = f"{item['code']} - {item['name']}"

    prefill = st.session_state.get("pending_question")
    if "pending_question" in st.session_state:
        del st.session_state["pending_question"]
    if prefill:
        st.session_state["ask_question"] = prefill

    left, right = st.columns([1.55, 1], gap="large")

    with left:
        question = st.text_area(
            "Research question",
            key="ask_question",
            height=150,
            placeholder="e.g. How does flame spread over thin fuels in microgravity? Cite the evidence.",
        )
        st.caption("Examples")
        example_columns = st.columns(2)
        for index, (label, text) in enumerate(EXAMPLES):
            with example_columns[index % 2]:
                st.button(
                    label,
                    key=f"ask_example_{index}",
                    use_container_width=True,
                    on_click=_set_question,
                    args=(text,),
                )
        with st.expander("Run options"):
            user_id = st.text_input("User ID", value="streamlit_user", key="ask_user_id")
            session_id = st.text_input("Session ID (optional)", value="", key="ask_session_id")
            include_reviewer = st.checkbox(
                "Run the scientific reviewer",
                value=True,
                key="ask_include_reviewer",
            )
            st.caption("Sessions are in-memory only; a new ADK session is created for every run.")
        selected = st.selectbox(
            "Workflow",
            options=options,
            format_func=lambda code: labels.get(code, code),
            index=0,
            key="ask_workflow",
        )
        run = st.button("Run IGNIS", type="primary", use_container_width=True)

    with right:
        plan_code = selected
        if selected == "auto" and question.strip():
            try:
                plan_code = api.classify_question(question)
            except Exception:
                plan_code = None
        plan = next((item for item in catalog if item["code"] == plan_code), None)
        theme.section_title("Workflow plan")
        if plan:
            plan_layers = api.summarize_layers(plan.get("layers") or [])
            plan_name = f"{plan['code']} / {plan['name']}"
            st.markdown(f"**{plan_name}**")
            st.caption(plan.get("use_case", ""))
            st.markdown(plan.get("description", ""))
            st.markdown(f"**Specialist layers:** {plan_layers}")
        else:
            theme.empty_state("Type a question to preview its route.")

        theme.divider()
        theme.section_title("Run history")
        history = st.session_state.get("run_history", [])
        if not history:
            theme.empty_state("No runs yet in this browser session.")
        for index, entry in enumerate(history):
            if st.button(entry["label"], key=f"ask_history_{index}", use_container_width=True):
                st.session_state["last_result"] = entry["result"]
                st.rerun()
        if history:
            if st.button("Clear history", key="ask_clear_history", use_container_width=True):
                st.session_state["run_history"] = []
                if "last_result" in st.session_state:
                    del st.session_state["last_result"]
                st.rerun()

        theme.divider()
        ui.render_callout(
            "Latency and limits",
            [
                "A run can take 20-90+ seconds depending on model latency.",
                "If a specialist fails, the run is marked partial and the failure is shown.",
                "Run history lives only in this browser session.",
            ],
            "warn",
        )

    if run:
        if not question.strip():
            st.warning("Enter a research question first.")
        elif selected == "auto" and plan_code is None:
            st.warning("Could not classify this question. Choose a workflow manually.")
        else:
            workflow_code = None if selected == "auto" else selected
            with st.spinner("Routing specialists. This can take a minute..."):
                result = api.run_workflow(
                    question,
                    workflow_code=workflow_code,
                    user_id=user_id or "streamlit_user",
                    session_id=session_id.strip() or None,
                    include_reviewer=include_reviewer,
                )
            st.session_state["last_result"] = result
            if "run_history" not in st.session_state:
                st.session_state["run_history"] = []
            history = st.session_state["run_history"]
            workflow_used = result.get("workflow", "?")
            label = f"{api.now_label()} - {workflow_used} - {question[:52]}"
            history.insert(0, {"label": label, "result": result})
            del history[8:]

    result = st.session_state.get("last_result")
    if result:
        theme.divider()
        ui.render_result(result, key_prefix="ask")
        ui.render_json_block(result, "ignis_workflow_result.json", "Full result JSON")
