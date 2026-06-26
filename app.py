"""
app.py
------
Streamlit chat UI for the textbook RAG chatbot.
Retrieves relevant chunks from ChromaDB and asks Groq's LLM to answer
using only that retrieved context.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

DB_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"  # fast + strong free-tier model on Groq
TOP_K = 4  # number of chunks to retrieve per question

st.set_page_config(page_title="Semester Textbook Chatbot", page_icon="📚")
st.title("📚 Semester Textbook Chatbot")
st.caption("Ask questions about your textbooks — answers are grounded in the actual book content.")


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if not os.path.exists(DB_DIR):
        return None
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


@st.cache_resource
def load_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0.2)


def build_prompt(question, retrieved_docs):
    context_blocks = []
    for i, doc in enumerate(retrieved_docs, 1):
        book = doc.metadata.get("book", "Unknown source")
        page = doc.metadata.get("page", "?")
        context_blocks.append(f"[Source {i}: {book}, page {page}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a helpful study assistant. Answer the student's question using ONLY the textbook excerpts provided below. If the excerpts don't contain enough information to answer, say so honestly instead of guessing.

Be clear and concise, like you're explaining to a fellow student. Use simple language and concrete examples where helpful.

TEXTBOOK EXCERPTS:
{context}

STUDENT QUESTION: {question}

ANSWER:"""
    return prompt


# --- Load resources ---
vectordb = load_vectorstore()
llm = load_llm()

if vectordb is None:
    st.error(
        "No knowledge base found. Run `python ingest.py` first after putting your "
        "textbook PDFs in the `books/` folder."
    )
    st.stop()

if llm is None:
    st.error(
        "GROQ_API_KEY not found. Create a `.env` file in this folder with:\n\n"
        "GROQ_API_KEY=your_key_here\n\n"
        "Get a free key at https://console.groq.com/keys"
    )
    st.stop()

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
question = st.chat_input("Ask a question about your textbooks...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching textbooks..."):
            retrieved_docs = vectordb.similarity_search(question, k=TOP_K)
            prompt = build_prompt(question, retrieved_docs)
            response = llm.invoke(prompt)
            answer = response.content

        st.markdown(answer)

        with st.expander("📖 Sources used"):
            for doc in retrieved_docs:
                book = doc.metadata.get("book", "Unknown")
                page = doc.metadata.get("page", "?")
                st.markdown(f"**{book}**, page {page}")
                st.caption(doc.page_content[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This chatbot answers questions using only your uploaded textbook PDFs. "
        "It retrieves the most relevant passages and asks an LLM to answer based on them"
        " currently trained on game theory book book ,  Financial Engineering Book"
    )
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()
