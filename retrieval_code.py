import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# load embedding model

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)
# load faiss index

index = faiss.read_index("faiss_index.index")

# load metadata

with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# user query

query = input("\nAsk Question: ")

# query format

query_for_embedding = (
    "Represent this sentence for searching relevant passages: "
    + query
)

# create query embedding

query_embedding = model.encode(
    query_for_embedding,
    normalize_embeddings=True
)

# Convert to numpy float32
query_embedding = np.array([query_embedding]).astype("float32")

# search top 5 query results

scores, indices = index.search(query_embedding, 5)

# print result 

print("\n" + "=" * 100)
print("TOP RETRIEVAL RESULTS")
print("=" * 100)

for rank, idx in enumerate(indices[0]):

    # Skip invalid index
    if idx == -1:
        continue

    result = metadata[idx]

    print(f"\nRESULT {rank + 1}")
    print("-" * 80)

    print("\nSECTION:")
    print(result["section"])

    print("\nURL:")
    print(result["url"])

    print("\nHEADINGS:")
    print(result["headings"])

    print("\nCHUNK:")
    print(result["chunk"])

    print("\nSIMILARITY SCORE:")
    print(scores[0][rank])

    print("\n" + "=" * 100)