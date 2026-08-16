import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

sentences = [
    "I love Artificial Intelligence",
    "Machine Learning is amazing",
    "Deep Learning is a subset of AI",
    "Neural Networks power modern AI",
    "I like pizza",
    "Football is fun"
]

embeddings = model.encode(
    sentences
).astype("float32")

d = embeddings.shape[1]