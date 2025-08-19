from transformers import pipeline
HF_ZERO_SHOT_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33"
zero_shot = pipeline("zero-shot-classification", model=HF_ZERO_SHOT_MODEL)

INTENTS = ["service_request","hotel_info","internal_experience","external_experience","booking","off_topic"]

def probs_intent(text: str, labels=INTENTS) -> dict:
  res = zero_shot(text, candidate_labels=labels, multi_label=False)
  # res["labels"] desc by score
  return {lab: float(score) for lab, score in zip(res["labels"], res["scores"])}

def predict_intent_with_hf(text: str, labels=INTENTS):
  p = probs_intent(text, labels)
  label = max(p, key=p.get)
  return {"intent": label, "confidence": p[label], "probs": p}
