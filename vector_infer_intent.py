import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.read_index("intent_faiss.index")

with open("intent_texts.csv", "r", encoding="utf-8") as f:
    texts = [l.strip() for l in f.readlines()[1:]]  # Skip header
with open("intent_labels.csv", "r", encoding="utf-8") as f:
    labels = [l.strip() for l in f.readlines()[1:]]

def predict_intent_knn(sentence, k=1):
    emb = model.encode([sentence]).astype("float32")
    D, I = faiss_index.search(emb, k)
    votes = {}
    for idx in I[0]:
        intent = labels[idx]
        votes[intent] = votes.get(intent, 0) + 1
    pred = max(votes, key=votes.get)
    return pred, [labels[i] for i in I[0]], [texts[i] for i in I[0]], D[0][0]

# Example usage
if __name__ == "__main__":
    query = "Can you call me a cab to the airport?"
    pred, neighbors, sents = predict_intent_knn(query)
    print(f"PREDICTED INTENT: {pred}")
    print("Top neighbors/labels:", list(zip(neighbors, sents)))
