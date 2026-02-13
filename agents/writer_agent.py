import json
from typing import Dict, Any, List, Optional
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_CHAT_MODEL = "qwen2.5:7b-instruct"


def _build_evidence_context(evidence: List[Dict[str, Any]], max_items: int = 8) -> str:
    """Build evidence context with better error handling."""
    if not evidence:
        return ""

    ev = evidence[:max_items]
    blocks = []
    for i, e in enumerate(ev, start=1):
        # More robust citation building
        citation = e.get("citation")
        if not citation:
            parts = []
            if e.get("doc_name"):
                parts.append(str(e["doc_name"]))
            if e.get("page"):
                parts.append(f"page {e['page']}")
            if e.get("section"):
                parts.append(f"section {e['section']}")
            if e.get("idx"):
                parts.append(f"chunk {e['idx']}")
            citation = " | ".join(parts) if parts else "Unknown source"

        text = (e.get("text") or "").strip()
        if text:  # Only include if there's actual text
            blocks.append(f"[E{i}] {citation}\n{text}")

    return "\n\n".join(blocks)


def _ollama_chat(
        prompt: str,
        model: str = DEFAULT_OLLAMA_CHAT_MODEL,
        timeout: int = 120,
        max_retries: int = 2
) -> str:
    """Call Ollama API with retry logic and better error handling."""
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
                                "Follow the schema exactly. Never invent facts. "
                                "If evidence is missing, output exactly: Not found in sources."
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
            break  # Don't retry connection errors
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.error(f"Request failed: {e}")
            break

    raise RuntimeError(f"Failed to get response from Ollama after {max_retries} attempts: {last_error}")


def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    """Safely parse JSON with multiple fallback strategies."""
    if not s:
        return None

    s = s.strip()

    # Try direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Try removing markdown code blocks
    if s.startswith("```"):
        lines = s.split("\n")
        if len(lines) > 2:
            s = "\n".join(lines[1:-1])  # Remove first and last lines
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass

    # Try finding JSON object boundaries
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start: end + 1])
        except json.JSONDecodeError:
            pass

    return None


def write_deliverable(
        user_task: str,
        research_output: Dict[str, Any],
        chat_model: str = DEFAULT_OLLAMA_CHAT_MODEL,
) -> Dict[str, Any]:
    """
    Generate a client-ready deliverable from research output.

    Args:
        user_task: The original task description
        research_output: Output from ResearchAgent with 'status' and 'evidence'
        chat_model: Ollama model to use for generation

    Returns:
        Dict with status, deliverable, and metadata
    """
    # Validate input
    if not research_output or not isinstance(research_output, dict):
        return {
            "status": "error",
            "message": "Invalid research_output: must be a non-empty dict",
            "user_task": user_task,
            "deliverable": None,
        }

    if research_output.get("status") != "found":
        return {
            "status": "not_found",
            "message": "Not found in sources.",
            "user_task": user_task,
            "deliverable": None,
        }

    evidence = research_output.get("evidence") or []

    if not evidence:
        return {
            "status": "not_found",
            "message": "No evidence provided in research output.",
            "user_task": user_task,
            "deliverable": None,
        }

    evidence_context = _build_evidence_context(evidence)

    if not evidence_context:
        return {
            "status": "error",
            "message": "Could not build evidence context from provided evidence.",
            "user_task": user_task,
            "deliverable": None,
        }

    prompt = f"""
You must write a client-ready deliverable using ONLY the EVIDENCE provided below.

HARD CONSTRAINTS (must follow exactly):
- Do NOT use outside knowledge. Do NOT guess. Do NOT add medical advice beyond the evidence.
- Do NOT invent any dates, deadlines, timeframes, or relative time (e.g., "2 weeks from today").
  If a due date is not explicitly stated in the evidence, output EXACTLY: "Not found in sources."
- Do NOT invent person names. The "owner" field must be a ROLE only (e.g., "Clinic Lead", "Care Team", "Nurse Lead").
- Use ONLY evidence refs like "E1", "E2" in evidence_refs. Do not cite sections like "Step 1.7.1" unless it appears in the evidence text itself.
- Every action MUST include evidence_refs (1–3 items). If an action cannot be supported, do not include it.
- If any required detail is missing, write EXACTLY "Not found in sources." for that field.

OUTPUT FORMAT:
Return ONLY valid JSON with this exact schema (keys must match exactly; snake_case only):

{{
  "executive_summary": "max 150 words",
  "client_ready_email": {{
    "subject": "...",
    "body": "..."
  }},
  "action_list": [
    {{
      "action": "...",
      "owner": "...",
      "due_date": "Not found in sources.",
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
        logger.error(f"Failed to get LLM response: {e}")
        return {
            "status": "error",
            "message": f"Failed to communicate with LLM: {str(e)}",
            "user_task": user_task,
            "deliverable": None,
        }

    payload = _safe_json_load(raw)

    if payload is None:
        logger.error(f"Failed to parse JSON from LLM output: {raw[:500]}")
        return {
            "status": "error",
            "message": "Writer model did not return valid JSON.",
            "raw_output": raw[:1000],  # Limit raw output length
            "user_task": user_task,
            "deliverable": None,
        }

    # Build sources list from evidence (limit to max_items used)
    sources = []
    max_items = min(8, len(evidence))
    for i in range(1, max_items + 1):
        e = evidence[i - 1]
        citation = e.get("citation") or "Unknown source"
        sources.append({
            "evidence_ref": f"E{i}",
            "citation": citation
        })

    # Ensure payload has the sources
    payload["sources"] = sources

    return {
        "status": "ok",
        "user_task": user_task,
        "deliverable": payload,
    }


if __name__ == "__main__":
    # Example usage with mock data for testing
    mock_research_output = {
        "status": "found",
        "evidence": [
            {
                "citation": "Dementia Guidelines | page 5 | section 2.1 | chunk 1",
                "text": "Non-pharmacological interventions should be first-line for agitation.",
                "doc_name": "Dementia Guidelines",
                "page": 5,
            }
        ]
    }

    task = "Summarize guideline-based management approaches for agitation in dementia."

    try:
        out = write_deliverable(task, mock_research_output)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error(f"Error in main: {e}")