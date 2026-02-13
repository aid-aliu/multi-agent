import re
from typing import Dict, Any, List, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_E_REF_RE = re.compile(r"\bE(\d+)\b")


def _collect_evidence_refs(deliverable: Dict[str, Any]) -> Set[str]:
    """
    Recursively collect all evidence references (E1, E2, etc.) from the deliverable.

    Args:
        deliverable: The deliverable dict to scan

    Returns:
        Set of evidence reference strings like {"E1", "E2"}
    """
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
    """
    Generate the set of valid evidence references based on evidence list length.

    Args:
        evidence: List of evidence items

    Returns:
        Set of valid references like {"E1", "E2", "E3"}
    """
    if not evidence:
        return set()
    return {f"E{i}" for i in range(1, len(evidence) + 1)}


def _default_due_date() -> str:
    """Return the standard 'not found' message for missing due dates."""
    return "Not found in sources."


def verify_deliverable(
        writer_output: Dict[str, Any],
        research_output: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Verify that a deliverable follows all constraints and references valid evidence.

    Args:
        writer_output: Output from write_deliverable()
        research_output: Output from ResearchAgent

    Returns:
        Dict with verification status, issues, and deliverable
    """
    # Validate inputs
    if not isinstance(writer_output, dict):
        return {
            "status": "error",
            "message": "writer_output must be a dict",
            "issues": [],
        }

    if not isinstance(research_output, dict):
        return {
            "status": "error",
            "message": "research_output must be a dict",
            "issues": [],
        }

    # Check writer status
    if writer_output.get("status") != "ok":
        return {
            "status": "blocked",
            "message": "Writer output is not OK; cannot verify.",
            "writer_status": writer_output.get("status"),
            "issues": [],
        }

    # Check research status
    if research_output.get("status") != "found":
        return {
            "status": "blocked",
            "message": "No evidence found; deliverable must be 'Not found in sources.'",
            "issues": [],
        }

    # Extract deliverable
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

    # Check 1: Verify all evidence refs in deliverable are valid
    used_refs = _collect_evidence_refs(deliverable)
    bad_refs = sorted(r for r in used_refs if r not in valid_refs)
    if bad_refs:
        issues.append({
            "type": "invalid_evidence_ref",
            "detail": f"Deliverable references evidence not provided: {bad_refs}",
            "severity": "error",
        })

    # Check 2: Verify action_list structure and requirements
    action_list = deliverable.get("action_list")

    if action_list is None:
        issues.append({
            "type": "missing_action_list",
            "detail": "Deliverable missing 'action_list' field.",
            "severity": "error",
        })
        action_list = []  # Set to empty to avoid further errors
    elif not isinstance(action_list, list):
        issues.append({
            "type": "invalid_action_list_type",
            "detail": f"'action_list' must be a list, got {type(action_list).__name__}",
            "severity": "error",
        })
        action_list = []

    for i, action in enumerate(action_list, start=1):
        if not isinstance(action, dict):
            issues.append({
                "type": "invalid_action_type",
                "detail": f"Action #{i} is not a dict (got {type(action).__name__})",
                "severity": "error",
            })
            continue

        # Check evidence_refs field
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

        # Check for unknown evidence refs
        unknown = [r for r in refs if r not in valid_refs]
        if unknown:
            issues.append({
                "type": "unknown_evidence_refs",
                "detail": f"Action #{i} references unknown evidence_refs: {unknown}",
                "severity": "error",
            })

        # Check confidence value
        conf = (action.get("confidence") or "").lower()
        if conf not in {"high", "medium", "low"}:
            issues.append({
                "type": "bad_confidence_value",
                "detail": f"Action #{i} has invalid confidence='{action.get('confidence')}'. Must be high|medium|low.",
                "severity": "warning",
            })

        # Check/fix due_date
        due_date = action.get("due_date")
        if not due_date or (isinstance(due_date, str) and not due_date.strip()):
            action["due_date"] = _default_due_date()
            issues.append({
                "type": "missing_due_date",
                "detail": f"Action #{i} missing due_date; auto-fixed to 'Not found in sources.'",
                "severity": "warning",
            })

    # Check 3: Verify required top-level fields exist
    required_fields = ["executive_summary", "client_ready_email", "action_list", "sources"]
    for field in required_fields:
        if field not in deliverable:
            issues.append({
                "type": "missing_required_field",
                "detail": f"Deliverable missing required field: '{field}'",
                "severity": "error",
            })

    # Check 4: Verify client_ready_email structure
    email = deliverable.get("client_ready_email")
    if email:
        if not isinstance(email, dict):
            issues.append({
                "type": "invalid_email_structure",
                "detail": "client_ready_email must be a dict",
                "severity": "error",
            })
        else:
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

    # Determine final status
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
    import json

    # Mock data for testing without dependencies
    mock_research = {
        "status": "found",
        "evidence": [
            {"text": "Evidence 1", "citation": "Doc1"},
            {"text": "Evidence 2", "citation": "Doc2"},
        ]
    }

    # Valid deliverable
    mock_writer_ok = {
        "status": "ok",
        "deliverable": {
            "executive_summary": "Test summary with E1 reference",
            "client_ready_email": {
                "subject": "Test",
                "body": "Email body with E2"
            },
            "action_list": [
                {
                    "action": "Do something",
                    "owner": "Team Lead",
                    "due_date": "2024-12-31",
                    "confidence": "high",
                    "evidence_refs": ["E1", "E2"]
                }
            ],
            "sources": []
        }
    }

    # Invalid deliverable (bad refs, missing fields)
    mock_writer_bad = {
        "status": "ok",
        "deliverable": {
            "executive_summary": "Test with E99",  # Invalid ref
            "action_list": [
                {
                    "action": "Do something",
                    "owner": "Team Lead",
                    # Missing due_date
                    "confidence": "invalid_value",  # Invalid confidence
                    "evidence_refs": []  # Empty refs
                }
            ]
        }
    }

    print("=== Testing valid deliverable ===")
    result = verify_deliverable(mock_writer_ok, mock_research)
    print(json.dumps(result, indent=2))

    print("\n=== Testing invalid deliverable ===")
    result = verify_deliverable(mock_writer_bad, mock_research)
    print(json.dumps(result, indent=2))