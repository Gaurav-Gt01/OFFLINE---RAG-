from typing import List
import uuid
import chromadb
from llm_handler import embed_text

def get_or_create_collection(name: str):
    """
    Create or load a ChromaDB collection with the given name.
    """
    client = chromadb.Client()
    # Try to get existing collection, else create a new one
    try:
        collection = client.get_collection(name=name)
    except Exception:
        collection = client.create_collection(name=name)
    return collection

def upsert_documents(collection, texts: List[str], ids: List[str] = None):
    """
    Upsert (add or update) documents with embeddings into the ChromaDB collection.
    """
    # Generate IDs if not provided
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in texts]
    # Generate embeddings for each text
    embeddings = []
    for text in texts:
        emb = embed_text(text)
        embeddings.append(emb)
    # Perform upsert
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts)

def query_collection(collection, query_text: str, n_results: int = 3) -> List[str]:
    """
    Query the collection for documents relevant to the query_text.
    Returns a list of matching document strings.
    """
    # Generate embedding for query
    query_embedding = embed_text(query_text)
    # Query the collection
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    # results is a dict with 'documents' key
    docs = []
    for doc_list in results.get('documents', []):
        for doc in doc_list:
            docs.append(doc)
    return docs