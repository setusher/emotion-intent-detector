import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Step 1: Load your CSV ---
df = pd.read_csv("intent_dataset_450.csv")  # Use your filename

# --- Step 2: Compute embeddings ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

# --- Step 3: Initialise FAISS index and add embeddings ---
embeddings = np.array(embeddings).astype("float32")
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(embeddings)

# --- Step 4: Save index and metadata for later inference use ---
faiss.write_index(faiss_index, "intent_faiss.index")
df["text"].to_csv("intent_texts.csv", index=False)
df["intent"].to_csv("intent_labels.csv", index=False)
