import json
from typing import Dict, Any, List
import requests

DEFAULT_OLLAMA_CHAT_MODEL = "qwen2.5:7b-instruct"


def _ollama_chat(prompt: str, model: str = DEFAULT_OLLAMA_CHAT_MODEL, timeout: int = 120) -> str:
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a planner. Produce concise, actionable plans. Follow the schema exactly."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def _safe_json_load(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end + 1])
            except Exception:
                return None
        return None


def plan_task(user_task: str, model: str = DEFAULT_OLLAMA_CHAT_MODEL) -> Dict[str, Any]:
    prompt = f"""
Create an execution plan for a multi-agent workflow: Plan → Research → Draft → Verify → Deliver.

Return ONLY valid JSON with this schema:
{{
  "goal": "...",
  "research_questions": ["...", "..."],
  "deliverable_requirements": [
    "Executive Summary (max 150 words)",
    "Client-ready Email",
    "Action List (owner, due date, confidence)",
    "Sources and citations"
  ],
  "draft_outline": [
    "Executive Summary",
    "Client-ready Email",
    "Action List",
    "Sources"
  ],
  "success_criteria": [
    "Uses only retrieved evidence",
    "If evidence missing: 'Not found in sources.'",
    "Citations include DocumentName + page/chunk id"
  ]
}}

USER TASK:
{user_task}
""".strip()

    raw = _ollama_chat(prompt, model=model)
    plan = _safe_json_load(raw)

    if plan is None:
        return {
            "status": "error",
            "message": "Planner model did not return valid JSON.",
            "raw_output": raw,
        }

    return {
        "status": "ok",
        "user_task": user_task,
        "plan": plan,
    }


if __name__ == "__main__":
    task = "Turn dementia guideline evidence into a decision-ready deliverable for a clinic lead about managing agitation."
    out = plan_task(task)
    print(json.dumps(out, ensure_ascii=False, indent=2))
