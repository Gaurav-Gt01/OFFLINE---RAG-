import streamlit as st
from document_processor import process_pdf, process_docx, process_image, split_text
from vector_store import get_or_create_collection, upsert_documents
from llm_handler import answer_question


st.set_page_config(
    page_title="NiTRO - Offline RAG",
    layout="wide",
)


st.markdown(
    """
    <div style="text-align: center; margin-top: 50px;">
        <h1 style="font-size: 60px; color: #4B0082; font-weight: bold;">NiTRO</h1>
        <p style="font-size: 18px; color: #555;">Completely offline RAG system ingesting Multimodal data</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ----------------- Initialize State -----------------
if "collection" not in st.session_state:
    st.session_state.collection = get_or_create_collection("documents")
if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------- Upload Section -----------------
st.subheader("📂 Upload Documents")
col1, col2, col3 = st.columns(3)

with col1:
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
with col2:
    docx_file = st.file_uploader("Upload DOCX", type=["docx"])
with col3:
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if st.button("⚡ Process Documents"):
    text_data = ""
    if pdf_file is not None:
        try:
            text_data += process_pdf(pdf_file)
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
    if docx_file is not None:
        try:
            text_data += process_docx(docx_file)
        except Exception as e:
            st.error(f"Error processing DOCX: {e}")
    if image_file is not None:
        try:
            text_data += process_image(image_file)
        except Exception as e:
            st.error(f"Error processing image: {e}")

    if text_data:
        chunks = split_text(text_data)
        upsert_documents(st.session_state.collection, chunks)
        st.success("✅ Document(s) processed and added to the knowledge base.")
    else:
        st.warning("⚠️ Please upload a file to process.")


# ----------------- Chat Section -----------------
st.subheader("💬 Ask a Question")

# Show history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------- Chat Section -----------------
st.subheader("💬 Ask a Question")

# Show history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type your question and press Enter...")
if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get full RAG pipeline answer
    answer = answer_question(st.session_state.collection, user_input)

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Display assistant reply in chat bubble
    with st.chat_message("assistant"):
        st.markdown(f"### 🧾 Answer\n{answer}")
