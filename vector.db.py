import json
import faiss
import numpy as np

#load embedding file

with open("embedding2_output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# making empty lists to store all embeddings and metadata

all_embeddings = []

metadata = []
    
for item in data:

    section = item["section"]

    url = item["url"]

    headings = item["headings"]

    chunk_embeddings = item["chunk_embeddings"]

    # loop through each chunk embedding

    for ce in chunk_embeddings:
        # getting chunk and embedding
        chunk = ce["chunk"]

        embedding = ce["embedding"]

        # convert python list  to numpy array and then to float32 for faiss
        embedding = np.array(embedding).astype("float32")

        # normalize embedding for cosine similarity search

        embedding = embedding / np.linalg.norm(embedding)

        # store embedding
        all_embeddings.append(embedding)

        # store metadata
        metadata.append({
            "section": section,
            "url": url,
            "headings": headings,
            "chunk": chunk
        })

# create embedding matrix ( faiss requires a 2D array as it cannot store list of arrays )

embedding_matrix = np.array(all_embeddings).astype("float32")

# shape of the vector dimension 

dimension = embedding_matrix.shape[1]

print("\nVector Dimension:")
print(dimension)

# create faiss index for IP inner product ( cosine similarity search with normalized vectors )
index = faiss.IndexFlatIP(dimension) # create vector search database to quicky find similar vectors using inner product ( cosine similarity with normalized vectors )

# add vectors to faiss index

index.add(embedding_matrix)

# save faiss index 

faiss.write_index(index, "faiss_index.index")

# saving metadata to json file

with open("metadata.json", "w", encoding="utf-8") as f:

    json.dump(metadata, f, indent=2, ensure_ascii=False)

# output 

print("FAISS DATABASE CREATED")

print(f"\nTOTAL VECTORS:{index.ntotal}") 

print(f"\nVECTOR DIMENSION:{index.d}")

print(f"\nTOTAL METADATA ROWS:{len(metadata)}")


