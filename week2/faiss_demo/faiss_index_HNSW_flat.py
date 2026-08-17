import faiss
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Documents
sentences = [
    "I love Artificial Intelligence",
    "Machine Learning is amazing",
    "Deep Learning is a subset of AI",
    "Neural Networks power modern AI",
    "I like pizza",
    "Football is fun"
]

# Generate embeddings
embeddings = model.encode(sentences).astype("float32")

# Embedding dimension
d = embeddings.shape[1]

# Create HNSW Index
M = 16

index = faiss.IndexHNSWFlat(d, M)

# Optional: Improve graph quality
index.hnsw.efConstruction = 200

# Add vectors
index.add(embeddings)

print(f"Total vectors stored: {index.ntotal}")

# Query
query = "I like food"

query_embedding = model.encode([query]).astype("float32")

# Optional: Improve search quality
index.hnsw.efSearch = 50

# Search
distances, indices = index.search(query_embedding, k=2)

print("Indices:", indices)
print("Distances:", distances)

print("\nNearest Documents:")

for idx in indices[0]:
    print(sentences[idx])