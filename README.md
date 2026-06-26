# 📚 Semester Textbook Chatbot/RAG-Based Document Q&A Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions using your
own textbook PDFs. Built with LangChain + ChromaDB + Groq (Llama 3.3 70B, free tier).

## How it works

1. `ingest.py` reads your textbook PDFs, splits them into ~1000-character chunks,
   converts each chunk into a vector embedding, and stores them in a local
   ChromaDB database (`chroma_db/` folder).
2. `app.py` is a Streamlit chat app. When you ask a question, it:
   - Embeds your question
   - Finds the most similar chunks from your textbooks (semantic search)
   - Sends those chunks + your question to Groq's LLM
   - Returns an answer grounded in your actual books, with sources shown

## Setup

### 1. Install dependencies

```bash
cd textbook_chatbot
pip install -r requirements.txt --break-system-packages
```

(If you're using a conda env like your `cuda_env`, activate it first — no need
for `--break-system-packages` inside a conda/venv environment.)

### 2. Add your PDFs

Drop your textbook PDFs into the `books/` folder:

```
textbook_chatbot/
└── books/
    ├── subject1_textbook.pdf
    ├── subject2_textbook.pdf
    └── subject3_textbook.pdf
```

### 3. Get a free Groq API key

- Go to https://console.groq.com/keys
- Sign up (free) and create an API key

### 4. Create your `.env` file

Copy `.env.example` to `.env` and paste your key:

```bash
cp .env.example .env
```

Then edit `.env`:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 5. Build the knowledge base (run once per book set)

```bash
python ingest.py
```

This will take a few minutes for 600-page PDFs (mostly CPU-bound embedding —
your MX330 GPU isn't needed here, embeddings run fine on CPU).

You'll see progress like:
```
📖 Loading: operations_research.pdf
   → 612 pages loaded
✂️  Split into 2840 chunks total
💾 Building Chroma vector store...
```

### 6. Run the chatbot

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Ask questions like:
- "Explain the difference between LP and IP formulations"
- "What does chapter 3 say about queueing models?"
- "Summarize the assumptions behind the EOQ model"

## Adding more books later

Just drop new PDFs into `books/` and re-run `python ingest.py` — it will add
them to the existing database (it doesn't wipe old ones, since it checks for
an existing DB at `chroma_db/`). If you want to start completely fresh, delete
the `chroma_db/` folder first.

## Notes / things you may want to tweak

- **Chunk size** (`CHUNK_SIZE` in `ingest.py`): 1000 chars works well for prose-heavy
  textbooks. If your books have lots of formulas/tables, try smaller chunks (500-700).
- **TOP_K** (`app.py`): currently retrieves top 4 chunks per question. Increase to 6-8
  if answers feel like they're missing context, at the cost of slightly slower/more
  expensive calls.
- **Multiple subjects**: since every chunk is tagged with its source filename, you can
  later add a subject filter (e.g., a dropdown to restrict search to one book) — happy
  to add that if useful.
- **Groq rate limits**: the free tier has generous but real rate limits. If you hit
  them, Groq's docs list current limits per model.

## Possible extensions (good portfolio additions)

- Add a "subject" filter dropdown so students pick which book to query
- Show page numbers as clickable citations
- Add conversation memory (multi-turn follow-up questions)
- Deploy publicly via Streamlit Community Cloud so classmates can use it
