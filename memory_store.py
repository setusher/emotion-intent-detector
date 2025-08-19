# memory_store.py
import os, json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Literal
import numpy as np

# Embeddings
from sentence_transformers import SentenceTransformer

# ---- config ----
DB_PATH = os.getenv("MEMORY_FILE", "memory.jsonl")
EMB_MODEL_NAME = os.getenv("MEM_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# load emb model once
_emb_model = SentenceTransformer(EMB_MODEL_NAME)

# ---- types ----
EmotionLabel = Literal["sadness", "joy", "love", "anger", "fear", "surprise", "neutral"]
IntentLabel  = Literal["service_request", "hotel_info", "internal_experience",
                       "external_experience", "booking", "off_topic"]

@dataclass
class Example:
    text: str
    emotion: str
    intent: str
    tags: Dict[str, Any]
    added_at: str
    emb: List[float]  # unit-normalized vector

# ---- io helpers ----
def _load_all() -> List[Example]:
    if not os.path.exists(DB_PATH):
        return []
    out: List[Example] = []
    with open(DB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Example(**json.loads(line)))
    return out

def _save_one(ex: Example) -> None:
    with open(DB_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")

# ---- public api ----
def add_example(text: str, emotion: str, intent: str, tags: Dict[str, Any]) -> None:
    """
    Store a labeled example with its embedding. Call this when you're confident
    the labels are correct (e.g., after human feedback or high-confidence predictions).
    """
    # compute normalized embedding
    vec = _emb_model.encode(text, normalize_embeddings=True).tolist()
    ex = Example(
        text=text,
        emotion=emotion,
        intent=intent,
        tags=tags or {},
        added_at=datetime.utcnow().isoformat(),
        emb=vec,
    )
    _save_one(ex)

def top_k_similar(text: str, k: int = 3) -> List[Example]:
    """
    Return the k most similar stored examples using cosine similarity in embedding space.
    """
    allx = _load_all()
    if not allx:
        return []
    q = _emb_model.encode(text, normalize_embeddings=True)
    M = np.array([e.emb for e in allx], dtype=np.float32)  # (N, d)
    # cosine because both q and rows are normalized
    sims = M @ q  # (N,)
    idx = np.argsort(-sims)[:k]
    return [allx[i] for i in idx]

def label_dist(text: str, task: Literal["emotion", "intent"], k: int = 5) -> Dict[str, float]:
    """
    kNN label distribution: counts labels among top-k similar examples.
    """
    exs = top_k_similar(text, k=k)
    if not exs:
        return {}
    if task == "emotion":
        labels = [e.emotion for e in exs]
    else:
        labels = [e.intent for e in exs]
    # distribution
    counts: Dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    total = sum(counts.values()) or 1
    return {lab: counts[lab] / total for lab in counts}

# Optional: small helpers for debugging
def all_examples() -> List[Dict[str, Any]]:
    return [asdict(e) for e in _load_all()]

def clear_memory() -> None:
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
