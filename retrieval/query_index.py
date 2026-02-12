import json
import faiss
import numpy as np
import requests

from settings import (
    FAISS_DIR,
    OLLAMA_EMBED_MODEL,
    TOP_K,
)

# -----------------------------
# Query embedding
# -----------------------------

def embed_query(text: str) -> np.ndarray:
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": OLLAMA_EMBED_MODEL,
            "prompt": text,
        },
        timeout=60,
    )
    response.raise_for_status()
    emb = response.json()["embedding"]
    return np.array([emb], dtype="float32")


# -----------------------------
# FAISS retrieval
# -----------------------------

def query_index(query: str, top_k: int = TOP_K):
    index = faiss.read_index(str(FAISS_DIR / "index.faiss"))

    with open(FAISS_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadatas = json.load(f)

    qvec = embed_query(query)
    distances, indices = index.search(qvec, top_k)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        meta = metadatas[idx]
        results.append({
            "rank": rank,
            "distance": float(dist),
            "doc_name": meta["doc_name"],
            "page": meta["page"],
            "section": meta["section"],
        })

    return results


# -----------------------------
# Manual test
# -----------------------------

if __name__ == "__main__":
    query = "management of agitation in dementia patients"
    hits = query_index(query)

    for h in hits:
        print(h)
