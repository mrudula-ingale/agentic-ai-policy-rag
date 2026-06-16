from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ai_policy"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Default number of retrieved chunks.
DEFAULT_RETRIEVAL_K = 5

# Number of chunks retrieved from each selected source.
RETRIEVAL_K_PER_SOURCE: int = 5

COLLECTION_NAME = "ai_policy_docs"

PDF_FILES = {
    "eu_ai_act": "eu_ai_act.pdf",
    "gdpr": "gdpr.pdf",
    "nist_ai_rmf": "nist_ai_rmf.pdf",
}

# LLM settings.
GROQ_MODEL_NAME: str = "llama-3.1-8b-instant"
LLM_TEMPERATURE: float = 0.0
MAX_CONTEXT_CHARACTERS: int = 8000
