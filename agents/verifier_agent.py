import re
from typing import Dict, Any, List, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    if not evidence:
        return set()
    return {f"E{i}" for i in range(1, len(evidence) + 1)}


def _default_due_date() -> str:
    return "Not found in sources."


def _is_not_found_text(x: Any) -> bool:
    return isinstance(x, str) and x.strip().lower() == "not found in sources."


def verify_deliverable(
    writer_output: Dict[str, Any],
    research_output: Dict[str, Any],
) -> Dict[str, Any]:

    if not isinstance(writer_output, dict):
        return {"status": "error", "message": "writer_output must be a dict", "issues": []}

    if not isinstance(research_output, dict):
        return {"status": "error", "message": "research_output must be a dict", "issues": []}

    if writer_output.get("status") != "ok":
        return {
            "status": "blocked",
            "message": "Writer output is not OK; cannot verify.",
            "writer_status": writer_output.get("status"),
            "issues": [],
        }

    if research_output.get("status") != "found":
        return {
            "status": "blocked",
            "message": "No evidence found; deliverable must be 'Not found in sources.'",
            "issues": [],
        }

    deliverable = writer_output.get("deliverable")
    if not deliverable or not isinstance(deliverable, dict):
        return {
            "status": "error",
            "message": "writer_output missing valid 'deliverable' dict",
            "issues": [],
        }

    evidence = research_output.get("evidence") or []
    if not evidence:
        return {
            "status": "blocked",
            "message": "Evidence list is empty; cannot verify references.",
            "issues": [],
        }

    valid_refs = _valid_ref_set(evidence)
    issues: List[Dict[str, Any]] = []

    # --- REQUIRED FIELDS ---
    required_fields = ["executive_summary", "client_ready_email", "action_list", "sources"]
    for field in required_fields:
        if field not in deliverable:
            issues.append({
                "type": "missing_required_field",
                "detail": f"Deliverable missing required field: '{field}'",
                "severity": "error",
            })

    # --- EXEC SUMMARY CHECKS ---
    exec_summary = deliverable.get("executive_summary")
    if _is_not_found_text(exec_summary):
        issues.append({
            "type": "invalid_not_found_executive_summary",
            "detail": "Executive summary is 'Not found in sources.' despite evidence being available.",
            "severity": "error",
        })
    elif isinstance(exec_summary, str):
        wc = len(exec_summary.split())
        if wc > 150:
            issues.append({
                "type": "executive_summary_too_long",
                "detail": f"Executive summary has {wc} words (max 150).",
                "severity": "error",
            })
    else:
        issues.append({
            "type": "missing_or_invalid_executive_summary",
            "detail": "executive_summary must be a non-empty string.",
            "severity": "error",
        })

    # --- EMAIL CHECKS ---
    email = deliverable.get("client_ready_email")
    if not isinstance(email, dict):
        issues.append({
            "type": "invalid_email_structure",
            "detail": "client_ready_email must be a dict",
            "severity": "error",
        })
        email = {}

    subj = email.get("subject")
    body = email.get("body")

    if _is_not_found_text(subj):
        issues.append({
            "type": "invalid_not_found_email_subject",
            "detail": "Email subject is 'Not found in sources.' despite evidence being available.",
            "severity": "error",
        })
    if _is_not_found_text(body):
        issues.append({
            "type": "invalid_not_found_email_body",
            "detail": "Email body is 'Not found in sources.' despite evidence being available.",
            "severity": "error",
        })

    if "subject" not in email:
        issues.append({
            "type": "missing_email_subject",
            "detail": "client_ready_email missing 'subject' field",
            "severity": "warning",
        })
    if "body" not in email:
        issues.append({
            "type": "missing_email_body",
            "detail": "client_ready_email missing 'body' field",
            "severity": "warning",
        })

    # --- EVIDENCE REF CHECKS (GLOBAL) ---
    used_refs = _collect_evidence_refs(deliverable)
    bad_refs = sorted(r for r in used_refs if r not in valid_refs)
    if bad_refs:
        issues.append({
            "type": "invalid_evidence_ref",
            "detail": f"Deliverable references evidence not provided: {bad_refs}",
            "severity": "error",
        })

    # --- ACTION LIST CHECKS ---
    action_list = deliverable.get("action_list")

    if action_list is None:
        issues.append({
            "type": "missing_action_list",
            "detail": "Deliverable missing 'action_list' field.",
            "severity": "error",
        })
        action_list = []
    elif not isinstance(action_list, list):
        issues.append({
            "type": "invalid_action_list_type",
            "detail": f"'action_list' must be a list, got {type(action_list).__name__}",
            "severity": "error",
        })
        action_list = []

    # Enforce at least 1 action when evidence exists
    if isinstance(action_list, list) and len(action_list) == 0:
        issues.append({
            "type": "empty_action_list",
            "detail": "Evidence exists but action_list is empty. Must include at least 1 supported action.",
            "severity": "error",
        })

    for i, action in enumerate(action_list, start=1):
        if not isinstance(action, dict):
            issues.append({
                "type": "invalid_action_type",
                "detail": f"Action #{i} is not a dict (got {type(action).__name__})",
                "severity": "error",
            })
            continue

        refs = action.get("evidence_refs")
        if refs is None or (isinstance(refs, list) and len(refs) == 0):
            issues.append({
                "type": "missing_evidence_refs",
                "detail": f"Action #{i} ('{action.get('action', 'unnamed')}') has no evidence_refs.",
                "severity": "error",
            })
            continue

        if not isinstance(refs, list):
            issues.append({
                "type": "invalid_evidence_refs_type",
                "detail": f"Action #{i} evidence_refs must be a list, got {type(refs).__name__}",
                "severity": "error",
            })
            continue

        unknown = [r for r in refs if r not in valid_refs]
        if unknown:
            issues.append({
                "type": "unknown_evidence_refs",
                "detail": f"Action #{i} references unknown evidence_refs: {unknown}",
                "severity": "error",
            })

        conf = (action.get("confidence") or "").lower()
        if conf not in {"high", "medium", "low"}:
            issues.append({
                "type": "bad_confidence_value",
                "detail": f"Action #{i} has invalid confidence='{action.get('confidence')}'. Must be high|medium|low.",
                "severity": "warning",
            })

        due_date = action.get("due_date")
        if not due_date or (isinstance(due_date, str) and not due_date.strip()):
            action["due_date"] = _default_due_date()
            issues.append({
                "type": "missing_due_date",
                "detail": f"Action #{i} missing due_date; auto-fixed to 'Not found in sources.'",
                "severity": "warning",
            })

    # --- FINAL DECISION ---
    error_issues = [issue for issue in issues if issue.get("severity") == "error"]

    if error_issues:
        logger.warning(f"Verifier blocked deliverable with {len(error_issues)} error(s)")
        return {
            "status": "blocked",
            "message": f"Verifier blocked deliverable due to {len(error_issues)} error(s).",
            "issues": issues,
            "deliverable": deliverable,
        }

    if issues:
        logger.info(f"Verifier passed with {len(issues)} warning(s)")
        return {
            "status": "ok",
            "message": f"Verifier passed deliverable with {len(issues)} warning(s).",
            "issues": issues,
            "deliverable": deliverable,
        }

    logger.info("Verifier passed deliverable with no issues")
    return {
        "status": "ok",
        "message": "Verifier passed deliverable.",
        "issues": [],
        "deliverable": deliverable,
    }


if __name__ == "__main__":
    import json as _json

    mock_research = {
        "status": "found",
        "evidence": [{"text": "Evidence 1", "citation": "Doc1"}, {"text": "Evidence 2", "citation": "Doc2"}],
    }

    mock_writer_ok = {
        "status": "ok",
        "deliverable": {
            "executive_summary": "Test summary with E1 reference",
            "client_ready_email": {"subject": "Test", "body": "Email body with E2"},
            "action_list": [
                {
                    "action": "Do something",
                    "owner": "Team Lead",
                    "due_date": "2024-12-31",
                    "confidence": "high",
                    "evidence_refs": ["E1", "E2"],
                }
            ],
            "sources": [],
        },
    }

    mock_writer_bad = {
        "status": "ok",
        "deliverable": {
            "executive_summary": "Not found in sources.",
            "client_ready_email": {"subject": "Not found in sources.", "body": "Not found in sources."},
            "action_list": [],
            "sources": [],
        },
    }

    print("=== Testing valid deliverable ===")
    print(_json.dumps(verify_deliverable(mock_writer_ok, mock_research), indent=2))

    print("\n=== Testing invalid deliverable ===")
    print(_json.dumps(verify_deliverable(mock_writer_bad, mock_research), indent=2))
