from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
DATA_OUT_DIR = PROJECT_ROOT / 'data' / 'processed'

FAISS_DIR = DATA_OUT_DIR / 'faiss_index'
CHUNKS_JSONL = DATA_OUT_DIR / 'chunks.jsonl'

CHUNKING_MODE = 'dynamic_tokens'

CHUNK_MAX_TOKENS = 280
CHUNK_TOKEN_OVERLAP = 50
OLLAMA_EMBED_MODEL = "mxbai-embed-large:latest"

TOP_K = 6