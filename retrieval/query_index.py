import json
from typing import List, Dict, Any

import faiss
import numpy as np
import requests

from retrieval.settings import FAISS_DIR, OLLAMA_EMBED_MODEL, TOP_K


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


def query_index(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    index = load_index()
    metadatas = load_metadatas()

    qvec = embed_query(query)
    distances, indices = index.search(qvec, top_k)

    out = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx < 0:
            continue
        meta = metadatas[idx]
        out.append({
            "rank": rank,
            "idx": int(idx),
            "distance": float(dist),
            "doc_name": meta.get("doc_name"),
            "page": meta.get("page"),
            "section": meta.get("section"),
        })
    return out


if __name__ == "__main__":
    q = "management of agitation in dementia patients"
    for hit in query_index(q):
        print(hit)
