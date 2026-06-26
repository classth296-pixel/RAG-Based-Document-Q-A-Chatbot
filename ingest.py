"""
ingest.py
----------
Reads all PDF textbooks from the books/ folder, splits them into chunks,
embeds them, and stores them in a local ChromaDB vector database.

Run this ONCE (or whenever you add/change books):
    python ingest.py
"""

import os
import pdfplumber
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

BOOKS_DIR = "books"
DB_DIR = "chroma_db"

# Chunk size: ~1000 characters with 200 overlap works well for textbook prose.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Free local embedding model — runs fine on CPU, no API key needed.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdf_with_pdfplumber(path: str, filename: str):
    """
    Load a single PDF using pdfplumber, page by page.
    Some PDFs have malformed embedded fonts that crash text extraction on
    specific pages (a known pypdf/pdfplumber edge case). Rather than letting
    one bad page kill the whole book, we skip that page and keep going.
    """
    docs = []
    skipped_pages = []

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                skipped_pages.append(page_num)
                continue

            if text.strip():  # skip blank pages
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"book": filename, "page": page_num},
                    )
                )

    if skipped_pages:
        print(f"   Skipped {len(skipped_pages)} unreadable page(s): {skipped_pages}")

    return docs


def load_all_pdfs(books_dir: str):
    """Load every PDF in books_dir and tag each page with its source filename."""
    all_docs = []
    pdf_files = [f for f in os.listdir(books_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f" No PDF files found in '{books_dir}/'. Add your textbook PDFs there first.")
        return all_docs

    for filename in pdf_files:
        path = os.path.join(books_dir, filename)
        print(f"Loading: {filename}")
        try:
            docs = load_pdf_with_pdfplumber(path, filename)
        except Exception as e:
            print(f"   ❌ Failed to open '{filename}': {e}")
            print("      Skipping this file and continuing with the rest.")
            continue

        all_docs.extend(docs)
        print(f"   → {len(docs)} pages loaded")

    return all_docs


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks total")
    return chunks


def build_vector_store(chunks):
    print(f"🧠 Loading embedding model: {EMBEDDING_MODEL} (first run downloads it, ~80MB)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"💾 Building Chroma vector store at '{DB_DIR}/' ...")
    # Batch to avoid memory spikes on large books
    batch_size = 200
    vectordb = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if vectordb is None:
            vectordb = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=DB_DIR,
            )
        else:
            vectordb.add_documents(batch)
        print(f"   embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    print("✅ Done! Vector store saved to disk.")


if __name__ == "__main__":
    print("=== Step 1: Loading PDFs ===")
    documents = load_all_pdfs(BOOKS_DIR)

    if not documents:
        exit(1)

    print("\n=== Step 2: Chunking text ===")
    chunks = chunk_documents(documents)

    print("\n=== Step 3: Embedding + storing ===")
    build_vector_store(chunks)

    print("\n🎉 Ingestion complete. Now run: streamlit run app.py")
