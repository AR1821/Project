import faiss
import numpy as np

index = faiss.read_index("faiss_index.index")

# Total vectors
print("Total vectors:", index.ntotal)

# Vector dimension
print("Dimension:", index.d)

# reconstruct first vector
vector = index.reconstruct(0)

print(vector)
