from transformers import pipeline
HF_MODEL = "j-hartmann/emotion-english-distilroberta-base"  # stronger baseline
clf = pipeline("text-classification", model=HF_MODEL, return_all_scores=True)

def probs_emotion(text: str) -> dict:
  # returns dict label->score
  scores = clf(text)[0]
  return {d["label"].lower(): float(d["score"]) for d in scores}

def predict_with_hf(text: str):
  p = probs_emotion(text)
  label = max(p, key=p.get)
  return {"label": label, "confidence": p[label], "probs": p}
