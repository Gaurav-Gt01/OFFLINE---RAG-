# 📚 Offline Multimodal RAG System

This project is a **Retrieval-Augmented Generation (RAG) system** that works fully **offline** using [Ollama](https://ollama.ai/), [ChromaDB](https://www.trychroma.com/), and [Streamlit](https://streamlit.io/).  

It allows you to **upload documents (PDF, DOCX, Images)**, index their contents in a vector database, and then ask natural language questions in a **chat-style interface**. The system retrieves relevant chunks and generates grounded answers using a local Large Language Model (LLM).

---

## 🚀 Features
- **Multi-format ingestion**: Supports PDF, DOCX, and Images (via OCR).
- **Offline embeddings & LLM**: Uses Ollama models locally, no internet required.
- **Vector search**: Stores embeddings in ChromaDB for fast retrieval.
- **Chat interface**: Clean Streamlit UI with persistent chat history.
- **Context-grounded answers**: LLM only answers based on retrieved context.

