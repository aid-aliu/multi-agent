import re
from typing import Dict, Any, List, Set


_E_REF_RE = re.compile(r"\bE(\d+)\b")


def _collect_evidence_refs(deliverable: Dict[str, Any]) -> Set[str]:
    refs: Set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            for m in _E_REF_RE.finditer(x):
                refs.add(f"E{m.group(1)}")

    walk(deliverable)
    return refs


def _valid_ref_set(evidence: List[Dict[str, Any]]) -> Set[str]:
    return {f"E{i}" for i in range(1, len(evidence) + 1)}


def _default_due_date():
    return "Not found in sources."


def verify_deliverable(
    writer_output: Dict[str, Any],
    research_output: Dict[str, Any],
) -> Dict[str, Any]:
    if writer_output.get("status") != "ok":
        return {
            "status": "blocked",
            "message": "Writer output is not OK; cannot verify.",
            "writer_status": writer_output.get("status"),
        }

    if research_output.get("status") != "found":
        return {
            "status": "blocked",
            "message": "No evidence found; deliverable must be 'Not found in sources.'",
        }

    deliverable = writer_output["deliverable"]
    evidence = research_output.get("evidence") or []
    valid_refs = _valid_ref_set(evidence)

    issues: List[Dict[str, Any]] = []

    used_refs = _collect_evidence_refs(deliverable)
    bad_refs = sorted(r for r in used_refs if r not in valid_refs)
    if bad_refs:
        issues.append({
            "type": "invalid_evidence_ref",
            "detail": f"Deliverable references evidence not provided: {bad_refs}",
        })

    action_list = deliverable.get("action_list") or []
    for i, a in enumerate(action_list, start=1):
        refs = a.get("evidence_refs") or []
        if not refs:
            issues.append({
                "type": "missing_evidence_refs",
                "detail": f"Action #{i} has no evidence_refs.",
            })
            continue

        unknown = [r for r in refs if r not in valid_refs]
        if unknown:
            issues.append({
                "type": "unknown_evidence_refs",
                "detail": f"Action #{i} references unknown evidence_refs: {unknown}",
            })

        conf = (a.get("confidence") or "").lower()
        if conf not in {"high", "medium", "low"}:
            issues.append({
                "type": "bad_confidence_value",
                "detail": f"Action #{i} has invalid confidence='{a.get('confidence')}'. Must be high|medium|low.",
            })

        if not a.get("due_date"):
            a["due_date"] = _default_due_date()
            issues.append({
                "type": "missing_due_date",
                "detail": f"Action #{i} missing due_date; set to 'Not found in sources.'",
            })

    if issues:
        return {
            "status": "blocked",
            "message": "Verifier blocked deliverable due to issues.",
            "issues": issues,
            "deliverable": deliverable,
        }

    return {
        "status": "ok",
        "message": "Verifier passed deliverable.",
        "deliverable": deliverable,
    }


if __name__ == "__main__":
    import json
    from agents.research_agent import EvidenceStore, ResearchAgent
    from agents.writer_agent import write_deliverable
    from retrieval.settings import CHUNKS_JSONL

    store = EvidenceStore(CHUNKS_JSONL)
    researcher = ResearchAgent(store=store)

    task = "Summarize evidence-based management approaches for agitation in dementia and propose an action list for a clinic team."
    research = researcher.search("management of agitation in dementia patients")

    writer_out = write_deliverable(task, research)
    verified = verify_deliverable(writer_out, research)

    print(json.dumps(verified, ensure_ascii=False, indent=2))
