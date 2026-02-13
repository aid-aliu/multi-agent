import json
from typing import Dict, Any, List, Optional
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_CHAT_MODEL = "qwen2.5:7b-instruct"


def _ollama_chat(
        prompt: str,
        model: str = DEFAULT_OLLAMA_CHAT_MODEL,
        timeout: int = 120,
        max_retries: int = 2
) -> str:
    """
    Call Ollama API with retry logic and error handling.

    Args:
        prompt: User prompt
        model: Ollama model name
        timeout: Request timeout in seconds
        max_retries: Number of retry attempts

    Returns:
        Model response text

    Raises:
        RuntimeError: If all attempts fail
    """
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
                            "content": "You are a planner. Produce concise, actionable plans. Follow the schema exactly."
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
            logger.warning(f"Attempt {attempt + 1}/{max_retries} timed out")
        except requests.exceptions.ConnectionError as e:
            last_error = e
            logger.error(f"Connection error: {e}. Is Ollama running on localhost:11434?")
            break  # Don't retry connection errors
        except requests.exceptions.HTTPError as e:
            last_error = e
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            break
        except Exception as e:
            last_error = e
            logger.error(f"Unexpected error: {e}")
            break

    raise RuntimeError(
        f"Failed to get response from Ollama after {max_retries} attempts: {last_error}"
    )


def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON with fallback strategies.

    Args:
        s: String that may contain JSON

    Returns:
        Parsed dict or None if parsing fails
    """
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
            # Remove first and last lines
            s = "\n".join(lines[1:-1])
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


def _validate_plan(plan: Dict[str, Any]) -> List[str]:
    """
    Validate plan structure and return list of issues.

    Args:
        plan: Parsed plan dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    issues = []

    # Check required fields
    required_fields = [
        "goal",
        "research_questions",
        "deliverable_requirements",
        "draft_outline",
        "success_criteria"
    ]

    for field in required_fields:
        if field not in plan:
            issues.append(f"Missing required field: '{field}'")
        elif field != "goal" and not isinstance(plan.get(field), list):
            issues.append(f"Field '{field}' must be a list")
        elif field == "goal" and not isinstance(plan.get(field), str):
            issues.append(f"Field 'goal' must be a string")

    # Check list fields are non-empty
    if "research_questions" in plan:
        if not plan["research_questions"]:
            issues.append("'research_questions' cannot be empty")
        elif not all(isinstance(q, str) for q in plan["research_questions"]):
            issues.append("All items in 'research_questions' must be strings")

    if "deliverable_requirements" in plan:
        if not plan["deliverable_requirements"]:
            issues.append("'deliverable_requirements' cannot be empty")

    if "draft_outline" in plan:
        if not plan["draft_outline"]:
            issues.append("'draft_outline' cannot be empty")

    if "success_criteria" in plan:
        if not plan["success_criteria"]:
            issues.append("'success_criteria' cannot be empty")

    return issues


def plan_task(
        user_task: str,
        model: str = DEFAULT_OLLAMA_CHAT_MODEL,
        validate: bool = True
) -> Dict[str, Any]:
    """
    Generate an execution plan for a multi-agent workflow.

    The plan guides downstream agents through: Plan → Research → Draft → Verify → Deliver

    Args:
        user_task: User's natural language task description
        model: Ollama model to use for planning
        validate: Whether to validate plan structure

    Returns:
        Dict with:
        - status: "ok" or "error"
        - plan: Parsed plan dict (if successful)
        - message/raw_output: Error details (if failed)
        - validation_issues: List of issues (if validation enabled and issues found)
    """
    # Input validation
    if not user_task or not user_task.strip():
        return {
            "status": "error",
            "message": "Empty task provided.",
            "user_task": user_task,
        }

    logger.info(f"Planning task: {user_task[:100]}...")

    prompt = f"""
Create an execution plan for a multi-agent workflow: Plan → Research → Draft → Verify → Deliver.

Return ONLY valid JSON with this schema:
{{
  "goal": "Clear statement of what we're trying to accomplish",
  "research_questions": [
    "Specific question 1 to guide retrieval",
    "Specific question 2 to guide retrieval"
  ],
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

    # Call LLM
    try:
        raw = _ollama_chat(prompt, model=model)
    except Exception as e:
        logger.error(f"Failed to get LLM response: {e}")
        return {
            "status": "error",
            "message": f"Failed to communicate with LLM: {str(e)}",
            "user_task": user_task,
        }

    # Parse JSON response
    plan = _safe_json_load(raw)

    if plan is None:
        logger.error(f"Failed to parse JSON from LLM output: {raw[:500]}")
        return {
            "status": "error",
            "message": "Planner model did not return valid JSON.",
            "raw_output": raw[:1000],  # Limit length
            "user_task": user_task,
        }

    # Validate plan structure
    validation_issues = []
    if validate:
        validation_issues = _validate_plan(plan)
        if validation_issues:
            logger.warning(f"Plan validation failed: {validation_issues}")
            return {
                "status": "error",
                "message": "Plan structure validation failed.",
                "validation_issues": validation_issues,
                "plan": plan,
                "user_task": user_task,
            }

    logger.info(f"Successfully created plan with {len(plan.get('research_questions', []))} research questions")

    return {
        "status": "ok",
        "user_task": user_task,
        "plan": plan,
    }


if __name__ == "__main__":
    # Example 1: Valid task
    task = "Turn dementia guideline evidence into a decision-ready deliverable for a clinic lead about managing agitation."

    try:
        out = plan_task(task)
        print("=== PLAN OUTPUT ===")
        print(json.dumps(out, ensure_ascii=False, indent=2))

        if out["status"] == "ok":
            plan = out["plan"]
            print("\n=== PLAN SUMMARY ===")
            print(f"Goal: {plan.get('goal')}")
            print(f"\nResearch Questions ({len(plan.get('research_questions', []))}):")
            for i, q in enumerate(plan.get("research_questions", []), 1):
                print(f"  {i}. {q}")
            print(f"\nDeliverable Requirements ({len(plan.get('deliverable_requirements', []))}):")
            for req in plan.get("deliverable_requirements", []):
                print(f"  • {req}")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback

        traceback.print_exc()

    # Example 2: Empty task (should fail gracefully)
    print("\n\n=== TESTING EMPTY TASK ===")
    out = plan_task("")
    print(json.dumps(out, ensure_ascii=False, indent=2))