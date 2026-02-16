import json
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
import requests

from .settings import FAISS_DIR, OLLAMA_EMBED_MODEL, TOP_K


def embed_query(text: str) -> np.ndarray:
    r = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    emb = r.json()["embedding"]
    return np.array([emb], dtype="float32")


def load_index():
    return faiss.read_index(str(FAISS_DIR / "index.faiss"))


def load_metadatas() -> List[Dict[str, Any]]:
    with open(FAISS_DIR / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


def infer_domain(doc_name: str) -> str:
    n = (doc_name or "").lower()

    if "adhd" in n or "attention-deficit" in n or "hyperactivity" in n:
        return "adhd"
    if "dementia" in n or "alzheim" in n:
        return "dementia"
    if "autism" in n or "asd" in n:
        return "autism"
    if "psychosis" in n or "schiz" in n or "sign145" in n or "sign-145" in n:
        return "psychosis"
    return "other"


def query_index(
    query: str,
    top_k: int = TOP_K,
    allowed_domains: Optional[List[str]] = None,
    allowed_doc_ids: Optional[List[str]] = None,
    overfetch: int = 50,
) -> List[Dict[str, Any]]:
    index = load_index()
    metadatas = load_metadatas()

    qvec = embed_query(query)

    k = max(int(overfetch), int(top_k))
    distances, indices = index.search(qvec, k)

    out = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < 0:
            continue

        meta = metadatas[int(idx)]
        doc_name = meta.get("doc_name")
        doc_id = meta.get("doc_id") or doc_name

        domain = meta.get("domain")
        if not domain:
            domain = infer_domain(doc_name)

        if allowed_domains and domain not in allowed_domains:
            continue
        if allowed_doc_ids and doc_id not in allowed_doc_ids:
            continue

        out.append(
            {
                "rank": len(out) + 1,
                "idx": int(idx),
                "distance": float(dist),
                "doc_name": doc_name,
                "doc_id": doc_id,
                "domain": domain,
                "page": meta.get("page"),
                "section": meta.get("section"),
            }
        )

        if len(out) >= int(top_k):
            break

    return out


if __name__ == "__main__":
    q = "adult ADHD non-pharmacological treatment"

    print("\n--- FILTERED (ADHD ONLY) ---")
    for hit in query_index(q, top_k=6, allowed_domains=["adhd"]):
        print(hit)

    print("\n--- UNFILTERED (MIXED) ---")
    for hit in query_index(q, top_k=6):
        print(hit)
