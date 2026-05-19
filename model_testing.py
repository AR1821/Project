import faiss
import json
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# load open source LLM for answer generation

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=513,
    temperature=0.7,
    do_sample=True
)

# load embedding model  

embedding_model = SentenceTransformer(
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

# query embedding

query_embedding = embedding_model.encode(
    query_for_embedding,
    normalize_embeddings=True
)

query_embedding = np.array([query_embedding]).astype("float32")
 
# search top 3 query results

scores, indices = index.search(query_embedding, 3)

# retrived   chunks and their metadata

retrieved_chunks = []

print("TOP RETRIEVED CHUNKS")

for rank, idx in enumerate(indices[0]):

    if idx == -1:
        continue

    result = metadata[idx]

    chunk = result["chunk"]

    retrieved_chunks.append(chunk)

    print(f"\nRESULT {rank + 1}")
    print("-" * 80)

    print("\nURL:")
    print(result["url"])

    print("\nCHUNK:")
    print(chunk)

    print("\nSIMILARITY SCORE:")
    print(scores[0][rank])

# combine retrieved chunks to create context for answer generation

context = "\n\n".join(retrieved_chunks)

# create prompt for answer generation using retrieved context and user query

prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY using the context below.

If answer is not in context, say:
"I could not find the answer in the provided context."
# CONTEXT
{context}

# QUESTION

{query}

#ANSWER
"""

# generate response from LLM using the prompt

response = generator(
    prompt,
    return_full_text=False
)

answer = response[0]["generated_text"]

# final answer output

print("\n" + "=" * 100)
print("FINAL ANSWER")
print("=" * 100)

print("\n")
print(answer)