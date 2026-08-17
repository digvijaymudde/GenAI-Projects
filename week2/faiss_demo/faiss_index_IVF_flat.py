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

d = 384

nlist = 1
quanatizer = faiss.IndexFlatL2(d)

index = faiss.IndexIVFFlat(quanatizer,d,nlist)

index.train(embeddings)
index.add(embeddings)
index.nprobe = 1

query = "I like food"

embed_query = model.encode(
    [query]
).astype("float32")

distance, indices = index.search(
    embed_query,
    k=2
)

print(indices)
print(distance)

print(sentences[indices[0][0]])
print(sentences[indices[0][1]])

