import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import run

# Configuration
QUESTIONS_PATH = Path(PROJECT_ROOT) / "eval" / "questions.jsonl"
OUT_DIR = Path(PROJECT_ROOT) / "eval" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_questions() -> List[Dict[str, Any]]:
    """Load evaluation questions from JSONL file."""
    if not QUESTIONS_PATH.exists():
        return []

    qs = []
    try:
        with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    qs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    st.warning(f"Skipping malformed line {line_num}: {e}")
    except Exception as e:
        st.error(f"Error loading questions: {e}")

    return qs


def has_citations(result: Optional[Dict[str, Any]]) -> bool:
    """Check if result contains valid citations."""
    if not result:
        return False
    sources = result.get("sources") or []
    return isinstance(sources, list) and len(sources) > 0


def says_not_found(result: Optional[Dict[str, Any]]) -> bool:
    """Check if result explicitly says 'not found in sources'."""
    if not result:
        return False

    r = result
    es = (r.get("executive_summary") or "").lower()
    email = r.get("client_ready_email") or {}
    subj = (email.get("subject") or "").lower()
    body = (email.get("body") or "").lower()

    return "not found in sources" in es or "not found in sources" in subj or "not found in sources" in body


def trace_visible(out: Dict[str, Any]) -> bool:
    """Check if trace log is present and non-empty."""
    t = out.get("trace")
    return isinstance(t, list) and len(t) > 0


def exec_summary_ok(result: Optional[Dict[str, Any]]) -> bool:
    """Check if executive summary exists and is ≤150 words."""
    if not result:
        return False

    es = result.get("executive_summary") or ""
    es = es.strip()

    if not es:
        return False

    word_count = len(es.split())
    return word_count <= 150


def run_eval() -> tuple[int, List[Dict[str, Any]], str]:
    """
    Run evaluation on all questions in eval/questions.jsonl.

    Returns:
        (failures_count, summary_list, report_path)
    """
    qs = load_questions()

    if not qs:
        st.error("No questions to evaluate")
        return 0, [], ""

    summary = []
    failures = 0

    progress = st.progress(0)
    status_box = st.empty()

    for i, q in enumerate(qs, start=1):
        question_id = q.get("id", f"q_{i}")
        task = q.get("task", "")

        status_box.write(f"Running {question_id} ({i}/{len(qs)})...")

        t0 = time.time()
        try:
            out = run(task)
        except Exception as e:
            st.error(f"Error running {question_id}: {e}")
            out = {
                "status": "error",
                "message": str(e),
                "result": None,
                "trace": []
            }

        sec = round(time.time() - t0, 2)

        # Save output
        out_path = OUT_DIR / f"{question_id}.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.warning(f"Failed to save output for {question_id}: {e}")

        # Run checks
        result = out.get("result") or {}
        checks = {
            "trace_visible": trace_visible(out),
            "exec_summary_le_150_words": exec_summary_ok(result),
            "has_citations_or_not_found": (has_citations(result) or says_not_found(result)),
            "status_ok_or_blocked": out.get("status") in {"ok", "blocked"},
        }

        passed = all(checks.values())
        if not passed:
            failures += 1

        summary.append({
            "id": question_id,
            "task": task,  # Store full task, not truncated
            "seconds": sec,
            "pass": passed,
            "status": out.get("status"),
            "checks": checks,
            "output_file": str(out_path),
        })

        progress.progress(i / len(qs))

    # Save summary report
    report_path = OUT_DIR / "summary.json"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "failures": failures,
                "total": len(qs),
                "pass_rate": f"{((len(qs) - failures) / len(qs) * 100):.1f}%",
                "summary": summary
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed to save summary report: {e}")
        report_path = OUT_DIR / "summary_failed.json"

    status_box.success("✅ Evaluation complete!")
    return failures, summary, str(report_path)


def render_deliverable(result: Dict[str, Any]) -> None:
    """Render the deliverable in a nice format."""
    if not result:
        st.warning("No deliverable to display.")
        return

    # Executive Summary
    st.markdown("#### 📋 Executive Summary")
    es = result.get("executive_summary", "")
    if es:
        st.info(es)
        word_count = len(es.split())
        if word_count > 150:
            st.warning(f"⚠️ Executive summary exceeds 150 words ({word_count} words)")
        else:
            st.caption(f"✓ {word_count} words (within 150 word limit)")
    else:
        st.info("Not provided")

    # Email
    st.markdown("#### 📧 Client-ready Email")
    email = result.get("client_ready_email") or {}
    subject = email.get("subject", "")
    body = email.get("body", "")

    if subject or body:
        if subject:
            st.markdown(f"**Subject:** {subject}")
        if body:
            with st.expander("Email body", expanded=True):
                st.text(body)
    else:
        st.info("Not provided")

    # Action List
    st.markdown("#### ✅ Action List")
    actions = result.get("action_list") or []

    if not actions:
        st.info("No actions.")
    else:
        for i, a in enumerate(actions, start=1):
            with st.expander(f"Action {i}: {a.get('action', 'Unnamed action')[:80]}...", expanded=False):
                st.markdown(f"**Full action:** {a.get('action', 'Not specified')}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Owner:** {a.get('owner', 'Not specified')}")
                with col2:
                    st.write(f"**Due date:** {a.get('due_date', 'Not specified')}")
                with col3:
                    confidence = a.get('confidence', 'unknown')
                    emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence.lower(), "⚪")
                    st.write(f"**Confidence:** {emoji} {confidence}")

                refs = a.get('evidence_refs', [])
                if refs:
                    st.write(f"**Evidence refs:** {', '.join(refs)}")

    # Sources
    st.markdown("#### 📚 Sources")
    sources = result.get("sources") or []

    if not sources:
        st.info("No sources provided.")
    else:
        with st.expander(f"View all citations ({len(sources)} total)", expanded=False):
            for s in sources:
                ref = s.get('evidence_ref', 'N/A')
                citation = s.get('citation', 'No citation')
                st.markdown(f"**{ref}:** {citation}")


def render_trace(trace: List[Dict[str, Any]]) -> None:
    """Render agent trace log in a readable format."""
    if not trace:
        st.info("No trace available")
        return

    for i, entry in enumerate(trace, start=1):
        agent = entry.get("agent", "Unknown")
        status = entry.get("status", "unknown")
        duration_ms = entry.get("duration_ms", 0)

        # Status emoji
        status_emoji = {
            "ok": "✅",
            "found": "✅",
            "not_found": "⚠️",
            "blocked": "🚫",
            "error": "❌"
        }.get(status, "⚪")

        with st.expander(f"{status_emoji} Step {i}: {agent} ({status}, {duration_ms}ms)", expanded=False):
            st.json(entry)


# ===== STREAMLIT UI =====

st.set_page_config(
    page_title="Enterprise Multi-Agent Copilot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Enterprise Multi-Agent Copilot")
st.caption("Plan → Research → Draft → Verify → Deliver")

# Sidebar
with st.sidebar:
    st.markdown("## Settings")
    mode = st.radio("Mode", ["💬 Chat", "📊 Eval (10 questions)"], index=0)

    st.markdown("---")
    st.markdown("## Display Options")
    show_trace = st.checkbox("Show trace log", value=True)
    show_raw = st.checkbox("Show raw JSON", value=False)

    st.markdown("---")
    st.markdown("## About")
    st.caption("Multi-agent system with:")
    st.caption("• Planner Agent")
    st.caption("• Research Agent")
    st.caption("• Writer Agent")
    st.caption("• Verifier Agent")

# Main content
if mode.startswith("💬"):  # Chat mode

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    prompt = st.chat_input("Type your task (e.g., 'Summarize dementia management guidelines')")

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Running multi-agent workflow..."):
                try:
                    t0 = time.time()
                    out = run(prompt)
                    elapsed_ms = int((time.time() - t0) * 1000)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback

                    with st.expander("Full error trace"):
                        st.code(traceback.format_exc())
                    st.stop()

            # Display status
            status = out.get("status", "unknown")
            status_color = {
                "ok": "🟢",
                "blocked": "🟡",
                "not_found": "⚠️",
                "error": "🔴"
            }.get(status, "⚪")

            st.markdown(f"**Status:** {status_color} {status}")

            # Display message if present
            message = out.get("message")
            if message:
                st.info(message)

            # Render deliverable
            result = out.get("result")
            if result:
                render_deliverable(result)
            else:
                st.warning("⚠️ No deliverable returned.")

            # Show trace
            if show_trace:
                st.markdown("---")
                st.markdown("### 🔍 Agent Trace")
                render_trace(out.get("trace", []))

            # Show runtime
            st.markdown("---")
            runtime_ms = out.get("total_ms", elapsed_ms)
            st.markdown(f"**⏱️ Runtime:** {runtime_ms} ms ({runtime_ms / 1000:.2f}s)")

            # Show raw JSON
            if show_raw:
                st.markdown("---")
                st.markdown("### 🔧 Raw JSON Output")
                st.json(out)

        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Status: {status}\n\nCheck deliverable above ↑"
        })

else:  # Eval mode
    st.markdown("### 📊 Evaluation Mode")
    st.caption("Runs eval/questions.jsonl (10 questions) and writes outputs to eval/results/")

    # Check if questions file exists
    if not QUESTIONS_PATH.exists():
        st.error(f"❌ Missing: {QUESTIONS_PATH}")
        st.info("Create this file with your evaluation questions in JSONL format.")
        st.code('''{"id": "q1", "task": "Your task here"}
{"id": "q2", "task": "Another task"}''', language="json")
        st.stop()

    # Display configuration
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Questions:** `{QUESTIONS_PATH}`")
    with col2:
        st.info(f"**Output dir:** `{OUT_DIR}`")

    # Preview questions
    questions = load_questions()
    st.markdown(f"**Found {len(questions)} questions**")

    if questions:
        with st.expander("Preview questions"):
            for i, q in enumerate(questions, start=1):
                st.markdown(f"**{i}. {q.get('id')}**")
                st.text(q.get('task', 'No task'))
                if i < len(questions):
                    st.markdown("---")

    # Run button
    if st.button("▶️ Run evaluation now", type="primary", disabled=len(questions) == 0):
        with st.spinner("Running evaluation..."):
            failures, summary, report_path = run_eval()

        # Display results
        st.markdown("---")
        st.markdown("### 📊 Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", len(summary))
        with col2:
            passed = len(summary) - failures
            st.metric("Passed", passed, delta=f"{(passed / len(summary) * 100):.1f}%")
        with col3:
            st.metric("Failed", failures, delta=f"-{(failures / len(summary) * 100):.1f}%" if failures > 0 else "0%")

        st.success(f"✅ Report saved to: `{report_path}`")

        # Show per-question results with ACTUAL DELIVERABLES
        st.markdown("### 📋 Per-question Results")

        for item in summary:
            passed = item["pass"]
            icon = "✅" if passed else "❌"

            with st.expander(f"{icon} {item['id']} - {item['status']} ({item['seconds']}s)", expanded=True):
                # Show the question
                st.markdown("#### 📝 Task")
                st.info(item.get('task', 'N/A'))

                # Show checks
                st.markdown("#### ✓ Checks")
                checks = item.get("checks", {})
                for check_name, check_passed in checks.items():
                    check_icon = "✅" if check_passed else "❌"
                    check_label = check_name.replace("_", " ").title()
                    st.markdown(f"{check_icon} {check_label}")

                # Load and display the actual deliverable
                st.markdown("---")
                st.markdown("#### 📄 Deliverable")
                try:
                    with open(item['output_file'], 'r') as f:
                        full_output = json.load(f)

                    result = full_output.get("result", {})
                    if result:
                        render_deliverable(result)

                        # Show trace if available
                        if show_trace and trace_visible(full_output):
                            st.markdown("---")
                            st.markdown("#### 🔍 Agent Trace")
                            render_trace(full_output.get("trace", []))

                        # Show raw JSON if requested
                        if show_raw:
                            st.markdown("---")
                            st.markdown("#### 🔧 Raw JSON")
                            st.json(full_output)
                    else:
                        st.warning("No deliverable in output")

                except Exception as e:
                    st.error(f"Failed to load output: {e}")

                st.caption(f"Output file: `{item['output_file']}`")

    # Show results folder
    st.markdown("---")
    st.markdown("### 📁 Results Folder")
    st.code(str(OUT_DIR), language="text")

    if st.button("🔄 Clear results"):
        import shutil

        try:
            shutil.rmtree(OUT_DIR)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            st.success("✅ Results cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear results: {e}")