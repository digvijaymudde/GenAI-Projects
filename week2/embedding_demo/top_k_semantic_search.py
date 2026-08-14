from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

#Query Embeddings 
query = "What is AI?"
query_embedding = model.encode([query])
print(type(embeddings))

#Finding score
scores = cosine_similarity(
    query_embedding,
    embeddings
)



top_k = 3
top_indices = np.argsort(scores[0])[::-1][:top_k]  
        #sort arrays indexs, reverse array, cut array with 3 length

for idx in top_indices:
    print(sentences[idx], scores[0][idx])