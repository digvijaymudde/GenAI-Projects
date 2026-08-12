from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

# Sentence embeddings
sentences = [
    "I love Artificial Intelligence",
    "Machine Learning is amazing",
    "I like pizza",
    "Football is fun"
]
embeddings = model.encode(sentences)

#Query Embeddings 
query = "What is AI?"
query_embedding = model.encode([query])
print(type(embeddings))

#Finding score
scores = cosine_similarity(
    query_embedding,
    embeddings
)

print(scores*100)

best_idx = scores.argmax()

print("Best Match:")
print(sentences[best_idx])

print("Score:")
print(scores[0][best_idx])