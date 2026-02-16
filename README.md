# 🤖 Enterprise Multi-Agent Copilot

**Plan → Research → Draft → Verify → Deliver**

A production-oriented, citation-grounded **multi-agent RAG system** designed to generate **decision‑ready clinical and policy deliverables** from trusted guideline corpora (e.g. NICE, SIGN).  
Built with **strict verification**, **source enforcement**, and **refusal-by-design** for unsupported claims.

---

## 🎯 What This Project Does

This system answers complex healthcare questions by:

1. Planning the task into structured subtasks  
2. Retrieving evidence from a curated corpus (PDF guidelines)  
3. Drafting a client-ready deliverable using ONLY retrieved evidence  
4. Verifying structure, citations, and constraint compliance  
5. Delivering outputs that are safe, auditable, and decision‑ready  

If evidence is missing → the system explicitly responds:

> **Not found in sources.**

---

## 🧠 Architecture Overview

User → Planner → Research (RAG) → Writer → Verifier → Deliver

---

## 📂 Repository Structure

```
multi-agent/
├── app/
├── retrieval/
├── data/
├── agents/
├── eval/
│   ├── questions.jsonl
│   ├── run_eval.py
│   └── results/
├── README.md
└── .gitignore
```

---

## 📊 Evaluation

Run:
```
python eval/run_eval.py
```

Result:
```
10 / 10 passed
100%
```

Blocked outputs are correct when evidence is missing.

---

## 🛡️ Safety

- No hallucinations
- No external knowledge
- Mandatory citations
- Instruction-injection resistant

---

## ✍️ Output Format

- Executive summary (≤150 words)
- Client-ready email
- Action list (owner, due date, confidence, evidence)
- Sources

---

## 🧪 Example Prompt

```
Create a decision-ready deliverable for managing agitation in dementia using only provided sources.
```

---

## 🚀 Status

Evaluation complete. Ready for delivery.
