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

dimentions = 384
indexx = faiss.IndexFlatL2(dimentions)
indexx.add(embeddings)
print(indexx.ntotal)

embedding_query = model.encode(["I love playing"])

eucli_distance, embed_index = indexx.search(embedding_query,k=2)
print(f"nearest vectors haivng index {embed_index} having distance of {eucli_distance}")
print(f"Nearest documents are {sentences[embed_index[0][0]] } \n {sentences[embed_index[0][1]] }")