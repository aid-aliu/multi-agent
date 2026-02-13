import json
import time

from agents.planner_agent import plan_task
from agents.research_agent import EvidenceStore, ResearchAgent
from agents.writer_agent import write_deliverable
from agents.verifier_agent import verify_deliverable
from retrieval.settings import CHUNKS_JSONL


def _ms():
    return int(time.time() * 1000)


def run(user_task: str):
    trace = []
    t0 = _ms()

    # 1) PLAN
    s = _ms()
    planner_out = plan_task(user_task)
    trace.append({"agent": "planner", "status": planner_out.get("status"), "ms": _ms() - s})
    if planner_out.get("status") != "ok":
        return {"status": "error", "where": "planner", "planner_out": planner_out, "trace": trace}

    plan = planner_out["plan"]
    research_qs = plan.get("research_questions") or [user_task]
    research_query = research_qs[0]

    # 2) RESEARCH
    s = _ms()
    store = EvidenceStore(CHUNKS_JSONL)
    researcher = ResearchAgent(store=store)
    research_out = researcher.search(research_query)
    trace.append({
        "agent": "research",
        "status": research_out.get("status"),
        "ms": _ms() - s,
        "query": research_query,
        "evidence_count": len(research_out.get("evidence") or []),
    })

    # If not found, stop early (required behavior)
    if research_out.get("status") != "found":
        return {
            "status": "ok",
            "result": {
                "executive_summary": "Not found in sources.",
                "client_ready_email": {"subject": "Not found in sources.", "body": "Not found in sources."},
                "action_list": [],
                "sources": [],
            },
            "trace": trace,
            "total_ms": _ms() - t0,
        }

    # 3) DRAFT (WRITER)
    s = _ms()
    writer_out = write_deliverable(user_task, research_out)
    trace.append({"agent": "writer", "status": writer_out.get("status"), "ms": _ms() - s})
    if writer_out.get("status") != "ok":
        return {"status": "error", "where": "writer", "writer_out": writer_out, "trace": trace}

    # 4) VERIFY
    s = _ms()
    verified = verify_deliverable(writer_out, research_out)
    trace.append({
        "agent": "verifier",
        "status": verified.get("status"),
        "ms": _ms() - s,
        "issues_count": len(verified.get("issues") or []),
    })

    return {
        "status": "ok" if verified.get("status") == "ok" else "blocked",
        "result": verified.get("deliverable"),
        "trace": trace,
        "total_ms": _ms() - t0,
    }


if __name__ == "__main__":
    task = input("Enter a task: ").strip()
    out = run(task)
    print(json.dumps(out, ensure_ascii=False, indent=2))
