import os
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

# ==========================
# Load Environment Variables
# ==========================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found")

# ==========================
# Documents
# ==========================

documents = [
    "Transformers use self attention.",
    "Embeddings convert text into vectors.",
    "FAISS is used for vector similarity search.",
    "HNSW is a graph based ANN algorithm."
]

# ==========================
# Load Embedding Model
# ==========================

model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================
# Create Document Embeddings
# ==========================

embeddings = model.encode(
    documents
).astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

# ==========================
# Create FAISS Index
# ==========================

d = embeddings.shape[1]

index = faiss.IndexFlatIP(d)

index.add(embeddings)

print(f"Total vectors stored: {index.ntotal}")

# ==========================
# User Query
# ==========================

query = "What is FAISS?"

query_embedding = model.encode(
    [query]
).astype("float32")

faiss.normalize_L2(query_embedding)

# ==========================
# Retrieve Top-K Chunks
# ==========================

k = 2

distances, indices = index.search(
    query_embedding,
    k
)

print("\nRetrieved Indices:")
print(indices)

retrieved_chunks = []

print("\nRetrieved Chunks:")

for idx in indices[0]:
    retrieved_chunks.append(documents[idx])
    print(documents[idx])

# ==========================
# Build Context
# ==========================

context = "\n".join(retrieved_chunks)

# ==========================
# Prompt Construction
# ==========================

prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{query}
"""

print("\nPrompt Sent To LLM:")
print(prompt)

# ==========================
# Call Groq
# ==========================

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",   # replace with a model available in your account
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0
)

# ==========================
# Final Answer
# ==========================

answer = response.choices[0].message.content

print("\nFinal Answer:")
print(answer)