import streamlit as st
import os
import sqlite3
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

# ----------------------
# Helper Functions
# ----------------------

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def create_faiss_vector_store(text, index_path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(index_path)
    return vector_store

def load_faiss_vector_store(index_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(index_path):
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    return None

def build_qa_chain(vector_store_path):
    vector_store = load_faiss_vector_store(vector_store_path)
    if vector_store is None:
        return None
    retriever = vector_store.as_retriever()
    llm = OllamaLLM(model="qwen3:4b")

    qa_chain = load_qa_chain(llm, chain_type="stuff")
    return RetrievalQA(retriever=retriever, combine_documents_chain=qa_chain)

# ----------------------
# Database Functions
# ----------------------

DB_PATH = "resources.db"

def search_resource_multi(question):
    words = question.lower().split()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    results = []
    for word in words:
        cur.execute("SELECT keyword, document_link, video_link FROM resources WHERE LOWER(keyword) LIKE ?", (f"%{word}%",))
        results.extend(cur.fetchall())
    conn.close()
    # Remove duplicates
    return list({(d[0], d[1], d[2]) for d in results})

# ----------------------
# Streamlit App
# ----------------------

st.title("🧑‍💼 Company Onboarding Chatbot")
st.write("Upload a PDF document for training or ask questions about company resources.")

# Initialize session state
if "doc_uploaded" not in st.session_state:
    st.session_state["doc_uploaded"] = False
    st.session_state["uploaded_file_name"] = None

# ----------------------
# Upload Document
# ----------------------
uploaded_file = st.file_uploader("Upload your document for training", type="pdf")

if uploaded_file:
    os.makedirs("uploaded", exist_ok=True)
    pdf_path = f"uploaded/{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {uploaded_file.name}")

    # Create FAISS index per document
    index_path = f"faiss_index_{uploaded_file.name}"
    text = extract_text_from_pdf(pdf_path)
    st.info(f"Creating FAISS vector store for {uploaded_file.name}...")
    create_faiss_vector_store(text, index_path)
    st.success("✅ Document is ready for QA!")

    # Update session state
    st.session_state["doc_uploaded"] = True
    st.session_state["uploaded_file_name"] = uploaded_file.name

# ----------------------
# Ask Question
# ----------------------
st.subheader("Ask questions or summarize your uploaded document")
question = st.text_input("Type your question here:")

if question:
    llm = OllamaLLM(model="qwen3:4b")

    if st.session_state.get("doc_uploaded", False):
        uploaded_file_name = st.session_state["uploaded_file_name"]
        index_path = f"faiss_index_{uploaded_file_name}"
        vector_store = load_faiss_vector_store(index_path)

        if vector_store is None:
            st.warning("Vector store not found. Please re-upload the document.")
        else:
            retriever = vector_store.as_retriever()
            qa_chain = build_qa_chain(index_path)

            if "summarize" in question.lower():
                # Retrieve all chunks
                all_docs = retriever.get_relevant_documents("")  # empty query returns all chunks
                full_text = "\n\n".join([doc.page_content for doc in all_docs])
                prompt = f"Summarize the following document:\n\n{full_text}"
                answer = llm(prompt=prompt)
                st.success(f"📄 Summary of {uploaded_file_name}:\n{answer}")
            else:
                answer = qa_chain.run(question)
                st.success(f"📄 Answer from {uploaded_file_name}:\n{answer}")
    else:
        # No document uploaded → Check DB first
        db_results = search_resource_multi(question)
        if db_results:
            st.success("Found resources in company DB:")
            for keyword, doc_path, video_link in db_results:
                if doc_path and os.path.exists(doc_path):
                    st.download_button(label=f"Download {keyword}", data=open(doc_path, "rb"), file_name=os.path.basename(doc_path))
                if video_link:
                    st.markdown(f"[Watch Video]({video_link})")
        else:
            # Fallback to external guidance
            answer = llm(prompt=question)
            st.success(f"💡 Answer:\n{answer}")