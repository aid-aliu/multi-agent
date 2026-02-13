import json
import time
import logging
from typing import Dict, Any, List

from agents.planner_agent import plan_task
from agents.research_agent import EvidenceStore, ResearchAgent
from agents.writer_agent import write_deliverable
from agents.verifier_agent import verify_deliverable
from retrieval.settings import CHUNKS_JSONL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


def _merge_research_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple research results into a single result.

    Args:
        results: List of research outputs

    Returns:
        Merged research output with combined evidence
    """
    if not results:
        return {
            "status": "not_found",
            "message": "No research results to merge",
            "evidence": [],
        }

    # If any result found evidence, consider it "found"
    found_results = [r for r in results if r.get("status") == "found"]

    if not found_results:
        # All searches failed
        return {
            "status": "not_found",
            "message": "Not found in sources.",
            "evidence": [],
        }

    # Combine evidence from all successful searches
    all_evidence = []
    for r in found_results:
        evidence = r.get("evidence") or []
        all_evidence.extend(evidence)

    # Deduplicate by idx (in case same chunk appears in multiple searches)
    seen_idx = set()
    unique_evidence = []
    for ev in all_evidence:
        idx = ev.get("idx")
        if idx not in seen_idx:
            seen_idx.add(idx)
            unique_evidence.append(ev)

    logger.info(f"Merged {len(results)} research results into {len(unique_evidence)} unique evidence items")

    return {
        "status": "found",
        "evidence": unique_evidence,
        "query_count": len(results),
    }


def run(user_task: str) -> Dict[str, Any]:
    """
    Execute the multi-agent workflow: Plan → Research → Draft → Verify → Deliver

    Args:
        user_task: Natural language task from user

    Returns:
        Dict with status, result (deliverable), trace, and metadata
    """
    if not user_task or not user_task.strip():
        logger.error("Empty task provided")
        return {
            "status": "error",
            "message": "Empty task provided",
            "trace": [],
            "total_ms": 0,
        }

    logger.info(f"Starting workflow for task: {user_task[:100]}...")

    trace = []
    t0 = _ms()

    # ===== STEP 1: PLAN =====
    logger.info("Step 1/4: Planning...")
    s = _ms()

    try:
        planner_out = plan_task(user_task)
    except Exception as e:
        logger.error(f"Planner failed: {e}")
        trace.append({
            "agent": "planner",
            "status": "error",
            "duration_ms": _ms() - s,
            "error": str(e),
        })
        return {
            "status": "error",
            "message": f"Planner failed: {str(e)}",
            "trace": trace,
            "total_ms": _ms() - t0,
        }

    trace.append({
        "agent": "planner",
        "status": planner_out.get("status"),
        "duration_ms": _ms() - s,
    })

    if planner_out.get("status") != "ok":
        logger.error(f"Planner returned non-ok status: {planner_out.get('status')}")
        return {
            "status": "error",
            "message": planner_out.get("message", "Planning failed"),
            "planner_output": planner_out,
            "trace": trace,
            "total_ms": _ms() - t0,
        }

    plan = planner_out.get("plan", {})
    research_questions = plan.get("research_questions") or [user_task]

    logger.info(f"Plan created with {len(research_questions)} research question(s)")

    # ===== STEP 2: RESEARCH =====
    logger.info("Step 2/4: Researching...")
    s = _ms()

    try:
        store = EvidenceStore(CHUNKS_JSONL)
    except Exception as e:
        logger.error(f"Failed to initialize evidence store: {e}")
        trace.append({
            "agent": "research",
            "status": "error",
            "duration_ms": _ms() - s,
            "error": str(e),
        })
        return {
            "status": "error",
            "message": f"Failed to load evidence store: {str(e)}",
            "trace": trace,
            "total_ms": _ms() - t0,
        }

    researcher = ResearchAgent(store=store)

    # Execute research queries (support multiple questions)
    research_results = []
    for i, query in enumerate(research_questions[:3], start=1):  # Limit to 3 queries
        logger.info(f"Research query {i}/{min(len(research_questions), 3)}: {query[:50]}...")
        try:
            result = researcher.search(query)
            research_results.append(result)
        except Exception as e:
            logger.error(f"Research query {i} failed: {e}")
            research_results.append({
                "status": "error",
                "message": str(e),
                "evidence": [],
            })

    # Merge results
    research_out = _merge_research_results(research_results)

    trace.append({
        "agent": "research",
        "status": research_out.get("status"),
        "duration_ms": _ms() - s,
        "queries": research_questions[:3],
        "evidence_count": len(research_out.get("evidence") or []),
    })

    # If not found, return early with "not found" deliverable
    if research_out.get("status") != "found":
        logger.info("No evidence found - returning 'not found' deliverable")
        return {
            "status": "ok",
            "message": "Not found in sources.",
            "result": {
                "executive_summary": "Not found in sources.",
                "client_ready_email": {
                    "subject": "Information Not Available",
                    "body": "Not found in sources."
                },
                "action_list": [],
                "sources": [],
            },
            "trace": trace,
            "total_ms": _ms() - t0,
        }

    # ===== STEP 3: DRAFT (WRITER) =====
    logger.info("Step 3/4: Writing deliverable...")
    s = _ms()

    try:
        writer_out = write_deliverable(user_task, research_out)
    except Exception as e:
        logger.error(f"Writer failed: {e}")
        trace.append({
            "agent": "writer",
            "status": "error",
            "duration_ms": _ms() - s,
            "error": str(e),
        })
        return {
            "status": "error",
            "message": f"Writer failed: {str(e)}",
            "trace": trace,
            "total_ms": _ms() - t0,
        }

    trace.append({
        "agent": "writer",
        "status": writer_out.get("status"),
        "duration_ms": _ms() - s,
    })

    if writer_out.get("status") != "ok":
        logger.error(f"Writer returned non-ok status: {writer_out.get('status')}")
        return {
            "status": "error",
            "message": writer_out.get("message", "Writing failed"),
            "writer_output": writer_out,
            "trace": trace,
            "total_ms": _ms() - t0,
        }

    # ===== STEP 4: VERIFY =====
    logger.info("Step 4/4: Verifying deliverable...")
    s = _ms()

    try:
        verified = verify_deliverable(writer_out, research_out)
    except Exception as e:
        logger.error(f"Verifier failed: {e}")
        trace.append({
            "agent": "verifier",
            "status": "error",
            "duration_ms": _ms() - s,
            "error": str(e),
        })
        return {
            "status": "error",
            "message": f"Verifier failed: {str(e)}",
            "trace": trace,
            "total_ms": _ms() - t0,
        }

    issues = verified.get("issues") or []
    trace.append({
        "agent": "verifier",
        "status": verified.get("status"),
        "duration_ms": _ms() - s,
        "issues_count": len(issues),
        "issues": issues if issues else None,  # Include issues for debugging
    })

    # Determine final status
    final_status = "ok" if verified.get("status") == "ok" else "blocked"

    total_time = _ms() - t0
    logger.info(f"Workflow complete: {final_status} ({total_time}ms)")

    return {
        "status": final_status,
        "message": verified.get("message"),
        "result": verified.get("deliverable"),
        "trace": trace,
        "total_ms": total_time,
    }


if __name__ == "__main__":
    import sys

    # Allow task from command line arg or stdin
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = input("Enter a task: ").strip()

    if not task:
        print("Error: No task provided")
        sys.exit(1)

    try:
        out = run(task)
        print("\n" + "=" * 60)
        print("WORKFLOW OUTPUT")
        print("=" * 60)
        print(json.dumps(out, ensure_ascii=False, indent=2))

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Status: {out.get('status')}")
        print(f"Total time: {out.get('total_ms')}ms")

        trace = out.get("trace", [])
        print(f"\nAgent execution:")
        for step in trace:
            agent = step.get("agent")
            status = step.get("status")
            duration = step.get("duration_ms")
            print(f"  {agent}: {status} ({duration}ms)")

        result = out.get("result")
        if result:
            print(f"\nDeliverable:")
            print(f"  Executive summary: {len(result.get('executive_summary', ''))} chars")
            print(f"  Actions: {len(result.get('action_list', []))}")
            print(f"  Sources: {len(result.get('sources', []))}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)