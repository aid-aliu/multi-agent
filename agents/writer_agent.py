import json
from typing import Dict, Any, List, Optional
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_CHAT_MODEL = "qwen2.5:7b-instruct"
NOT_FOUND = "Not found in sources."


def _build_citation(e: Dict[str, Any]) -> str:
    citation = e.get("citation")
    if citation:
        return str(citation)

    parts = []
    if e.get("doc_name"):
        parts.append(str(e["doc_name"]))
    if e.get("page"):
        parts.append(f"page {e['page']}")
    if e.get("section"):
        parts.append(f"section {e['section']}")
    if e.get("idx") is not None:
        parts.append(f"chunk {e['idx']}")
    return " | ".join(parts) if parts else "Unknown source"


def _build_evidence_context(evidence: List[Dict[str, Any]], max_items: int = 8) -> str:
    if not evidence:
        return ""

    ev = evidence[:max_items]
    blocks = []
    for i, e in enumerate(ev, start=1):
        text = (e.get("text") or "").strip()
        if not text:
            continue
        citation = _build_citation(e)
        blocks.append(f"[E{i}] {citation}\n{text}")

    return "\n\n".join(blocks)


def _ollama_chat(
    prompt: str,
    model: str = DEFAULT_OLLAMA_CHAT_MODEL,
    timeout: int = 120,
    max_retries: int = 2
) -> str:
    last_error = None

    for attempt in range(max_retries):
        try:
            r = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a strict, citation-grounded technical writer. "
                                "Use ONLY the provided evidence. "
                                "Return ONLY valid JSON that matches the schema exactly. "
                                "Never invent facts, names, dates, or timelines. "
                                f'Use "{NOT_FOUND}" ONLY for fields that cannot be supported by evidence.'
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                },
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            return data["message"]["content"]

        except requests.exceptions.Timeout as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} timed out")
        except requests.exceptions.ConnectionError as e:
            last_error = e
            logger.error(f"Connection error: {e}. Is Ollama running on localhost:11434?")
            break
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.error(f"Request failed: {e}")
            break

    raise RuntimeError(f"Failed to get response from Ollama after {max_retries} attempts: {last_error}")


def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None

    s = s.strip()

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    if s.startswith("```"):
        lines = s.split("\n")
        if len(lines) > 2:
            s2 = "\n".join(lines[1:-1])
            try:
                return json.loads(s2)
            except json.JSONDecodeError:
                pass

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _ensure_schema(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make schema robust (don’t fix truth; just enforce structure).
    Verifier will still block if evidence refs are missing/invalid.
    """
    if "executive_summary" not in payload:
        payload["executive_summary"] = NOT_FOUND

    if "client_ready_email" not in payload or not isinstance(payload["client_ready_email"], dict):
        payload["client_ready_email"] = {"subject": NOT_FOUND, "body": NOT_FOUND}
    else:
        payload["client_ready_email"].setdefault("subject", NOT_FOUND)
        payload["client_ready_email"].setdefault("body", NOT_FOUND)

    if "action_list" not in payload or not isinstance(payload["action_list"], list):
        payload["action_list"] = []

    if "sources" not in payload or not isinstance(payload["sources"], list):
        payload["sources"] = []

    for a in payload["action_list"]:
        if not isinstance(a, dict):
            continue
        a.setdefault("action", NOT_FOUND)
        a.setdefault("owner", "Clinic Lead")
        a.setdefault("due_date", NOT_FOUND)
        a.setdefault("confidence", "medium")
        a.setdefault("evidence_refs", [])

    return payload


def write_deliverable(
    user_task: str,
    research_output: Dict[str, Any],
    chat_model: str = DEFAULT_OLLAMA_CHAT_MODEL,
) -> Dict[str, Any]:

    if not research_output or not isinstance(research_output, dict):
        return {"status": "error", "message": "Invalid research_output", "user_task": user_task, "deliverable": None}

    if research_output.get("status") != "found":
        return {"status": "not_found", "message": NOT_FOUND, "user_task": user_task, "deliverable": None}

    evidence = research_output.get("evidence") or []
    if not evidence:
        return {"status": "not_found", "message": "No evidence provided in research output.", "user_task": user_task, "deliverable": None}

    evidence_context = _build_evidence_context(evidence)
    if not evidence_context:
        return {"status": "error", "message": "Could not build evidence context.", "user_task": user_task, "deliverable": None}

    prompt = f"""
You must write a client-ready deliverable using ONLY the EVIDENCE below.

KEY RULES:
- Use ONLY E1..E8 references for citations.
- Because evidence IS PROVIDED, you MUST NOT output "{NOT_FOUND}" for the entire executive_summary or the entire email.
  Use "{NOT_FOUND}" only for specific fields that are truly missing from evidence (e.g., due_date).
- Do NOT invent dates/timelines. If due date isn't explicitly in evidence, set due_date to "{NOT_FOUND}".
- Every action MUST include evidence_refs (1–3 items). If you can't support an action, don't include it.

OUTPUT: Return ONLY valid JSON with this exact schema:

{{
  "executive_summary": "max 150 words",
  "client_ready_email": {{
    "subject": "...",
    "body": "..."
  }},
  "action_list": [
    {{
      "action": "...",
      "owner": "ROLE ONLY",
      "due_date": "{NOT_FOUND}",
      "confidence": "high|medium|low",
      "evidence_refs": ["E1","E4"]
    }}
  ],
  "sources": [
    {{
      "evidence_ref": "E1",
      "citation": "DocumentName | page X | section Y | chunk Z"
    }}
  ]
}}

EVIDENCE:
{evidence_context}

USER TASK:
{user_task}
""".strip()

    try:
        raw = _ollama_chat(prompt, model=chat_model)
    except Exception as e:
        return {"status": "error", "message": f"Failed to communicate with LLM: {str(e)}", "user_task": user_task, "deliverable": None}

    payload = _safe_json_load(raw)
    if payload is None:
        return {"status": "error", "message": "Writer model did not return valid JSON.", "raw_output": raw[:1000], "user_task": user_task, "deliverable": None}

    payload = _ensure_schema(payload)

    # Build sources from the SAME evidence slice used in context
    sources = []
    max_items = min(8, len(evidence))
    for i in range(1, max_items + 1):
        e = evidence[i - 1]
        sources.append({"evidence_ref": f"E{i}", "citation": _build_citation(e)})

    payload["sources"] = sources

    return {"status": "ok", "user_task": user_task, "deliverable": payload}
