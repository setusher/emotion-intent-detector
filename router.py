import os
from typing import TypedDict
from free_metadata import tag_text_free
from memory_store import add_example, label_dist
from vector_infer_intent import predict_intent_knn
from hf_baseline import predict_with_hf, probs_emotion

# Import Gemini function!
from lc_gemini_intent import predict_intent_with_gemini

class Pred(TypedDict):
    label: str
    confidence: float
    source: str

class Out(TypedDict):
    emotion: Pred
    intent: Pred
    tags: dict

ALPHA_EMO = float(os.getenv("ALPHA_EMO","0.7"))
AUTO_STORE = os.getenv("MEM_AUTO_STORE","1") == "1"

def _blend(probs_a: dict, probs_b: dict, alpha: float) -> dict:
    keys = set(probs_a) | set(probs_b)
    out = {}
    for k in keys:
        pa, pb = probs_a.get(k, 0.0), probs_b.get(k, 0.0)
        out[k] = alpha*pa + (1-alpha)*pb
    s = sum(out.values()) or 1.0
    return {k: v/s for k,v in out.items()}

def predict(text: str) -> Out:
    # 1) HF + memory for emotion (as before)
    emo_model = predict_with_hf(text)
    emo_mem = label_dist(text, task="emotion", k=5)
    emo_blend = _blend(emo_model.get("probs", {}), emo_mem, ALPHA_EMO)
    emo_label = max(emo_blend, key=emo_blend.get) if emo_blend else emo_model["label"]
    emo_conf = float(emo_blend.get(emo_label, emo_model["confidence"]))

    # --- Vector DB for intent with fallback to Gemini ---
    int_label_knn, _, _, dist = predict_intent_knn(text)
    THRESHOLD = 0.9  # adjust if needed

    if dist > THRESHOLD:
        # Use Gemini fallback
        gemini_pred = predict_intent_with_gemini(text)
        int_label = gemini_pred.intent
        int_conf = float(gemini_pred.confidence)
        int_source = "gemini"
    else:
        int_label = int_label_knn
        int_conf = 1.0
        int_source = "vector_db"

    # Tag metadata (as before)
    action = {
        "service_request":"request_service",
        "hotel_info":"ask_info",
        "internal_experience":"ask_info",
        "external_experience":"ask_info",
        "booking":"book",
        "off_topic":"other"
    }.get(int_label, "other")
    tags = tag_text_free(text, action_from_intent=action)

    # Store confident examples (as before)
    if AUTO_STORE and emo_conf>=0.8 and int_conf>=0.8:
        try:
            add_example(text, emo_label, int_label, tags)
        except Exception as e:
            print("Memory store error:", repr(e))

    return {
        "emotion": {"label": emo_label, "confidence": emo_conf, "source": "hf+mem"},
        "intent":  {"label": int_label, "confidence": int_conf, "source": int_source},
        "tags": tags
    }
