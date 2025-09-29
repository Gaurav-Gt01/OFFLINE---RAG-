import ollama

# Ollama models
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "ictrek/llama3.2:3b"

def embed_text(text: str):
    """
    Generate embeddings using Ollama embedding model.
    """
    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )
    return response["embedding"]

def call_llm(context: str, query: str):
    """
    Generate a descriptive, well-structured answer using Ollama LLM.
    """
    prompt = f"""
You are a knowledgeable AI assistant. 
Answer the question in a clear, detailed, and descriptive way. 
Use only the context provided below. If the context does not contain enough information, say you don’t know. 
Avoid one-liners; explain step by step if possible.

Context:
{context}

Question:
{query}

Answer (detailed explanation):
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise, descriptive, and reliable AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


def answer_question(collection, query: str, n_results: int = 3):
    """
    Full RAG pipeline: embed query -> retrieve context -> call LLM.
    """
    from vector_store import query_collection  # lazy import to avoid circular dependency

    docs = query_collection(collection, query, n_results=n_results)
    context = "\n\n".join(docs) if docs else "No relevant context found."
    return call_llm(context, query)

