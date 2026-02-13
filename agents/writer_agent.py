import json
from typing import Dict, Any, List, Optional
import requests

DEFAULT_OLLAMA_CHAT_MODEL = "qwen2.5:7b-instruct"


def _build_evidence_context(evidence: List[Dict[str, Any]], max_items: int = 8) -> str:
    ev = evidence[:max_items]
    blocks = []
    for i, e in enumerate(ev, start=1):
        citation = e.get("citation") or f"{e.get('doc_name')} | page {e.get('page')} | section {e.get('section')} | chunk {e.get('idx')}"
        text = (e.get("text") or "").strip()
        blocks.append(f"[E{i}] {citation}\n{text}")
    return "\n\n".join(blocks)


def _ollama_chat(prompt: str, model: str = DEFAULT_OLLAMA_CHAT_MODEL, timeout: int = 120) -> str:
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise technical writer. Follow instructions exactly."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except Exception:
            return None
    return None


def write_deliverable(
    user_task: str,
    research_output: Dict[str, Any],
    chat_model: str = DEFAULT_OLLAMA_CHAT_MODEL,
) -> Dict[str, Any]:
    if research_output.get("status") != "found":
        return {
            "status": "not_found",
            "message": "Not found in sources.",
            "user_task": user_task,
            "deliverable": None,
        }

    evidence = research_output.get("evidence") or []
    evidence_context = _build_evidence_context(evidence)

    prompt = f"""
You must write a client-ready deliverable using ONLY the EVIDENCE provided below.
If a required detail is missing in the evidence, write "Not found in sources." for that detail.
Do not use outside knowledge. Do not guess. Do not add medical advice beyond the evidence.

Return ONLY valid JSON with this schema:
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
      "due_date": "...",
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

    raw = _ollama_chat(prompt, model=chat_model)
    payload = _safe_json_load(raw)

    if payload is None:
        return {
            "status": "error",
            "message": "Writer model did not return valid JSON.",
            "raw_output": raw,
        }

    sources = []
    for i, e in enumerate(evidence[:8], start=1):
        sources.append({"evidence_ref": f"E{i}", "citation": e.get("citation")})

    payload["sources"] = sources
    return {
        "status": "ok",
        "user_task": user_task,
        "deliverable": payload,
    }


if __name__ == "__main__":
    from agents.research_agent import EvidenceStore, ResearchAgent
    from retrieval.settings import CHUNKS_JSONL

    store = EvidenceStore(CHUNKS_JSONL)
    researcher = ResearchAgent(store=store)

    task = "Summarize evidence-based management approaches for agitation in dementia and propose an action list for a clinic team."
    research = researcher.search("management of agitation in dementia patients")

    out = write_deliverable(task, research)
    print(json.dumps(out, ensure_ascii=False, indent=2))
