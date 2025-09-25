from sentence_transformers import SentenceTransformer
import ollama

# HuggingFace embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Use your installed Ollama LLM
LLM_MODEL = "llama3.2:latest"
EMBED_MODEL = "nomic-embed-text:latest"  # not strictly needed since using HuggingFace for embeddings

def embed_text(text: str):
    """Generate embeddings using HuggingFace MiniLM"""
    return embedder.encode(text).tolist()

def call_llm(context: str, query: str):
    """Generate an answer using Ollama LLM, grounded in retrieved context"""
    prompt = f"""You are an AI assistant. Answer the question using only the context provided.

Context:
{context}

Question:
{query}

Answer:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]