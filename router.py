# router.py (free path)
import os
from typing import TypedDict
from free_metadata import tag_text_free
from memory_store import add_example, label_dist

from hf_baseline import predict_with_hf, probs_emotion
from hf_intent import predict_intent_with_hf, probs_intent, INTENTS as INTENT_LABELS

class Pred(TypedDict):
  label: str
  confidence: float
  source: str

class Out(TypedDict):
  emotion: Pred
  intent: Pred
  tags: dict

ALPHA_EMO = float(os.getenv("ALPHA_EMO","0.7"))    # weight for model vs memory
ALPHA_INT = float(os.getenv("ALPHA_INT","0.7"))
AUTO_STORE = os.getenv("MEM_AUTO_STORE","1") == "1"

def _blend(probs_a: dict, probs_b: dict, alpha: float) -> dict:
  # normalized convex combination
  keys = set(probs_a) | set(probs_b)
  out = {}
  for k in keys:
    pa, pb = probs_a.get(k, 0.0), probs_b.get(k, 0.0)
    out[k] = alpha*pa + (1-alpha)*pb
  # normalize
  s = sum(out.values()) or 1.0
  return {k: v/s for k,v in out.items()}

def predict(text: str) -> Out:
  # 1) HF models (free)
  emo_model = predict_with_hf(text)        # returns label, confidence, probs
  int_model = predict_intent_with_hf(text) # returns intent, confidence, probs

  # 2) Memory distributions (free)
  emo_mem = label_dist(text, task="emotion", k=5)  # dict label->prob
  int_mem = label_dist(text, task="intent",  k=5)

  # 3) Blend (model+memory)
  emo_blend = _blend(emo_model.get("probs", {}), emo_mem, ALPHA_EMO)
  int_blend = _blend(int_model.get("probs", {}), int_mem, ALPHA_INT)

  emo_label = max(emo_blend, key=emo_blend.get) if emo_blend else emo_model["label"]
  int_label = max(int_blend, key=int_blend.get) if int_blend else int_model["intent"]

  emo_conf = float(emo_blend.get(emo_label, emo_model["confidence"]))
  int_conf = float(int_blend.get(int_label, int_model["confidence"]))

  # 4) Free metadata (action from intent)
  action = {
    "service_request":"request_service",
    "hotel_info":"ask_info",
    "internal_experience":"ask_info",
    "external_experience":"ask_info",
    "booking":"book",
    "off_topic":"other"
  }.get(int_label, "other")
  tags = tag_text_free(text, action_from_intent=action)

  # 5) Store confident examples
  if AUTO_STORE and emo_conf>=0.8 and int_conf>=0.8:
    try:
      add_example(text, emo_label, int_label, tags)
    except Exception as e:
      print("Memory store error:", repr(e))

  return {
    "emotion": {"label": emo_label, "confidence": emo_conf, "source": "hf+mem"},
    "intent":  {"label": int_label, "confidence": int_conf, "source": "hf+mem"},
    "tags": tags
  }
