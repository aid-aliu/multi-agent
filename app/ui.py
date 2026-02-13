import os
import sys
import json
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import run


st.set_page_config(page_title="Enterprise Multi-Agent Copilot", layout="wide")
st.title("Enterprise Multi-Agent Copilot")
st.caption("Plan → Research → Draft → Verify → Deliver")

default_task = (
    "Create a decision-ready deliverable for a clinic lead on managing agitation in dementia "
    "using only the provided guidelines. Include (1) an executive summary (≤150 words), "
    "(2) a client-ready email, and (3) an action list with owner, due date, confidence, "
    "and citations for each action."
)

task = st.text_area("Task", value=default_task, height=140)

col_run, col_opts = st.columns([1, 1])

with col_opts:
    show_trace = st.checkbox("Show trace log", value=True)

run_btn = col_run.button("Run", type="primary")

if run_btn:
    with st.spinner("Running agents..."):
        out = run(task)

    st.subheader("Status")
    st.write(out.get("status"))

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Deliverable")
        result = out.get("result")
        if result is None:
            st.warning("No deliverable returned.")
        else:
            st.json(result)

    with right:
        if show_trace:
            st.subheader("Trace")
            st.json(out.get("trace", []))

        st.subheader("Runtime")
        st.write(f"{out.get('total_ms', 0)} ms")

    st.subheader("Raw output (debug)")
    st.code(json.dumps(out, ensure_ascii=False, indent=2), language="json")
