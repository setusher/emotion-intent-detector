# file: eval_quick.py
import random
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from router import predict

# dataset: 6 classes, columns: text, label
ds = load_dataset("dair-ai/emotion")
test = ds["test"]

id2label = test.features["label"].names  # ["sadness","joy","love","anger","fear","surprise"]

# sample a small subset to save cost
N = 200
idxs = random.sample(range(len(test)), N)

y_true, y_pred = [], []

for i in idxs:
    text = test[i]["text"]
    gold = id2label[test[i]["label"]]
    pred = predict(text)
    y_true.append(gold)
    y_pred.append(pred["label"])

print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=3))
