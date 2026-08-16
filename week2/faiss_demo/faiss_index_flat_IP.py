import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Sentence embeddings
sentences = [
    "I love Artificial Intelligence",
    "Machine Learning is amazing",
    "Deep Learning is a subset of AI",
    "Neural Networks power modern AI",
    "I like pizza",
    "Football is fun"
]
embeddings = model.encode(sentences)
print(embeddings[:5])
print(embeddings.shape)

faiss.normalize_L2(embeddings)
dimentions = 384
indexx = faiss.IndexFlatIP(dimentions)
indexx.add(embeddings)
print(indexx.ntotal)

embedding_query = model.encode(["I love playing"])
embedding_query = model.encode(
    ["I love playing"]
).astype("float32")

faiss.normalize_L2(embedding_query)

scores, indices = indexx.search(embedding_query,k=2)
print(f"nearest vectors haivng index {indices} having distance of {scores}")
print(f"Nearest documents are {sentences[indices[0][0]] } \n {sentences[indices[0][1]] }")