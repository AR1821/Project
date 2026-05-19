import json
from sentence_transformers import SentenceTransformer # used to generate embeddings

# load model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# load chunked file
with open("chunked2_output.json", "r", encoding="utf-8") as f:
    data = json.load(f) # load the chunked data from json file into python variable ( use of f )

final_output = []

# loop through each URL object
for item in data:

    section = item.get("section")
    url = item.get("url")
    headings = item.get("headings", []) # missing give default []
    chunks = item.get("chunks", [])

    chunk_embeddings = []

    # Generate embedding for each chunk
    for chunk in chunks:
        # converting chunk into embedding using the model and normalizing it for cosine similarity search
        embedding = model.encode(
            chunk,
            normalize_embeddings=True
        ).tolist() # convert numpy array to list for json serialization

        chunk_embeddings.append({
            "chunk": chunk,
            "embedding": embedding
        })

    # store one URL object
    final_output.append({
        "section": section,
        "url": url,
        "headings": headings,
        "total_chunks": len(chunks),
        "chunk_embeddings": chunk_embeddings
    })

# save embedding file
with open("embedding2_output.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

print(" Embeddings generated successfully!")