#!/usr/bin/env python3
"""
CLI Evaluation Runner for Enterprise Multi-Agent Copilot

Runs all questions in questions.jsonl and generates a detailed report.
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from app.main import run

# Configuration
QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_questions() -> List[Dict[str, Any]]:
    """Load evaluation questions from JSONL file with validation."""
    if not QUESTIONS_PATH.exists():
        print(f"ERROR: Questions file not found: {QUESTIONS_PATH}")
        sys.exit(1)

    qs = []
    try:
        with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    q = json.loads(line)
                    # Validate required fields
                    if "id" not in q:
                        print(f"WARNING: Line {line_num} missing 'id', skipping")
                        continue
                    if "task" not in q:
                        print(f"WARNING: Line {line_num} missing 'task', skipping")
                        continue
                    qs.append(q)
                except json.JSONDecodeError as e:
                    print(f"WARNING: Line {line_num} malformed JSON: {e}, skipping")
                    continue
    except Exception as e:
        print(f"ERROR: Failed to load questions: {e}")
        sys.exit(1)

    if not qs:
        print("ERROR: No valid questions found")
        sys.exit(1)

    return qs


def has_citations(result: Dict[str, Any]) -> bool:
    """Check if result contains valid citations."""
    if not result:
        return False
    sources = result.get("sources") or []
    return isinstance(sources, list) and len(sources) > 0


def says_not_found(result: Dict[str, Any]) -> bool:
    """Check if result explicitly says 'not found in sources'."""
    if not result:
        return False
    es = (result.get("executive_summary") or "").strip().lower()
    email = result.get("client_ready_email") or {}
    subj = (email.get("subject") or "").strip().lower()
    body = (email.get("body") or "").strip().lower()
    return "not found in sources" in es or "not found in sources" in subj or "not found in sources" in body


def trace_visible(out: Dict[str, Any]) -> bool:
    """Check if trace log is present and non-empty."""
    trace = out.get("trace")
    return isinstance(trace, list) and len(trace) > 0


def executive_summary_word_count_ok(result: Dict[str, Any]) -> bool:
    """Check if executive summary is ≤150 words."""
    if not result:
        return False
    es = (result.get("executive_summary") or "").strip()
    if not es:
        return False
    word_count = len(es.split())
    return word_count <= 150


def print_divider(char="=", length=80):
    """Print a divider line."""
    print(char * length)


def print_result_details(result: Dict[str, Any], indent=2):
    """Print the key parts of the deliverable."""
    prefix = " " * indent

    if not result:
        print(f"{prefix}❌ No result generated")
        return

    # Executive Summary
    print(f"{prefix}📋 EXECUTIVE SUMMARY:")
    es = result.get("executive_summary", "Not found")
    print(f"{prefix}   {es}")
    print()

    # Email
    email = result.get("client_ready_email", {})
    if email:
        print(f"{prefix}📧 CLIENT EMAIL:")
        print(f"{prefix}   Subject: {email.get('subject', 'N/A')}")
        body = email.get('body', 'N/A')
        if len(body) > 300:
            print(f"{prefix}   Body: {body[:300]}...")
        else:
            print(f"{prefix}   Body: {body}")
        print()

    # Action List
    actions = result.get("action_list", [])
    if actions:
        print(f"{prefix}✅ ACTION LIST ({len(actions)} items):")
        for i, action in enumerate(actions, 1):
            print(f"{prefix}   {i}. {action.get('action', 'N/A')}")
            print(f"{prefix}      Owner: {action.get('owner', 'N/A')}")
            print(f"{prefix}      Due: {action.get('due_date', 'N/A')}")
            print(f"{prefix}      Confidence: {action.get('confidence', 'N/A')}")
            print(f"{prefix}      Evidence: {action.get('evidence_refs', [])}")
        print()

    # Sources
    sources = result.get("sources", [])
    if sources:
        print(f"{prefix}📚 SOURCES ({len(sources)} citations):")
        for i, source in enumerate(sources[:5], 1):  # Show first 5
            print(f"{prefix}   [{source.get('evidence_ref', 'N/A')}] {source.get('citation', 'N/A')}")
        if len(sources) > 5:
            print(f"{prefix}   ... and {len(sources) - 5} more")
        print()


def evaluate_one(q: Dict[str, Any], verbose: bool = True) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluate a single question.

    Returns:
        (output_dict, info_dict) where info_dict contains checks and metadata
    """
    t0 = time.time()

    try:
        out = run(q["task"])
    except Exception as e:
        elapsed = time.time() - t0
        if verbose:
            print(f"  ❌ ERROR: {e}")
        return None, {
            "id": q["id"],
            "seconds": round(elapsed, 2),
            "error": str(e),
            "checks": {
                "trace_visible": False,
                "exec_summary_le_150_words": False,
                "has_citations_or_not_found": False,
                "status_ok_or_blocked": False,
            }
        }

    elapsed = time.time() - t0

    result = out.get("result") or {}
    checks = {
        "trace_visible": trace_visible(out),
        "exec_summary_le_150_words": executive_summary_word_count_ok(result),
        "has_citations_or_not_found": (has_citations(result) or says_not_found(result)),
        "status_ok_or_blocked": out.get("status") in {"ok", "blocked"},
    }

    return out, {
        "id": q["id"],
        "seconds": round(elapsed, 2),
        "status": out.get("status"),
        "checks": checks
    }


def main():
    """Main evaluation loop."""
    print_divider("=")
    print("Enterprise Multi-Agent Copilot - Evaluation Runner")
    print_divider("=")
    print()

    # Load questions
    print(f"📂 Loading questions from: {QUESTIONS_PATH}")
    questions = load_questions()
    print(f"✓ Loaded {len(questions)} questions\n")

    # Show output directory
    print(f"💾 Output directory: {OUT_DIR}")
    print()

    # Run evaluation
    print("🚀 Running evaluation...")
    print_divider("-")

    summary = []
    failures = 0
    errors = 0
    total_time = 0

    for i, q in enumerate(questions, start=1):
        qid = q["id"]
        task = q["task"]

        print()
        print_divider("=")
        print(f"[{i}/{len(questions)}] Question: {qid}")
        print_divider("=")
        print()
        print(f"📝 TASK:")
        print(f"   {task}")
        print()
        print(f"⏳ Running workflow...")

        out, info = evaluate_one(q, verbose=True)

        # Save output (even if there was an error)
        out_path = OUT_DIR / f"{qid}.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                if out:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                else:
                    json.dump({"error": info.get("error")}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  ⚠️  WARNING: Failed to save output: {e}")

        # Check if passed
        if out is None:
            errors += 1
            ok = False
        else:
            ok = all(info["checks"].values())
            if not ok:
                failures += 1

        total_time += info["seconds"]

        # Add to summary
        summary.append({
            "id": qid,
            "task": q["task"],
            "seconds": info["seconds"],
            "status": info.get("status"),
            "pass": ok,
            "checks": info["checks"],
            "output_file": str(out_path),
        })

        # Print result
        print()
        print_divider("-", 80)
        status_icon = "✅" if ok else "❌"
        status_text = "PASS" if ok else "FAIL"
        print(f"{status_icon} RESULT: {status_text} ({info['seconds']:.2f}s, status={info.get('status')})")
        print_divider("-", 80)
        print()

        # Show the actual deliverable
        if out:
            result = out.get("result", {})
            print_result_details(result)

        # Show trace summary
        if out and trace_visible(out):
            trace = out.get("trace", [])
            print(f"   🔍 TRACE ({len(trace)} steps):")
            for step in trace:
                agent = step.get("agent", "unknown")
                status = step.get("status", "unknown")
                print(f"      • {agent}: {status}")
            print()

        # Print failed checks
        if not ok:
            failed_checks = [k for k, v in info["checks"].items() if not v]
            print(f"   ⚠️  Failed checks: {', '.join(failed_checks)}")
            print()

    print_divider("=")
    print()

    # Save summary report
    report_path = OUT_DIR / "summary.json"
    try:
        pass_rate = ((len(questions) - failures - errors) / len(questions) * 100) if len(questions) > 0 else 0
        avg_time = total_time / len(questions) if len(questions) > 0 else 0

        report_data = {
            "total": len(questions),
            "passed": len(questions) - failures - errors,
            "failures": failures,
            "errors": errors,
            "pass_rate": f"{pass_rate:.1f}%",
            "total_time_seconds": round(total_time, 2),
            "avg_time_seconds": round(avg_time, 2),
            "summary": summary
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        print(f"✓ Report saved to: {report_path}")
    except Exception as e:
        print(f"✗ Failed to save report: {e}")

    # Print final summary
    print()
    print_divider("=")
    print("EVALUATION SUMMARY")
    print_divider("=")
    print(f"Total questions:  {len(questions)}")
    print(f"Passed:           {len(questions) - failures - errors} ✅ ({pass_rate:.1f}%)")
    print(f"Failed:           {failures} ❌")
    print(f"Errors:           {errors} ⚠️")
    print(f"Total time:       {total_time:.2f}s")
    print(f"Average time:     {avg_time:.2f}s")
    print_divider("=")

    # Exit with appropriate code
    if errors > 0 or failures > 0:
        print(f"\n⚠️  Evaluation completed with {failures} failure(s) and {errors} error(s)")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)