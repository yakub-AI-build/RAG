# Sample RAG Project (LangChain + Claude)

A minimal Retrieval-Augmented Generation (RAG) example. It indexes local
documents into a vector store, then answers questions using only the retrieved
content.

## How it works

1. **`ingest.py`** — loads files from `data/`, splits them into chunks, embeds
   them with a local `sentence-transformers` model, and stores them in a local
   Chroma database (`chroma_db/`).
2. **`query.py`** — embeds your question, retrieves the most relevant chunks,
   and asks Claude to answer from that context.

Embeddings run locally (no API cost). Only the final answer step calls the
Anthropic API.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then add your ANTHROPIC_API_KEY
```

## Usage

```bash
# 1. Build the index (re-run whenever documents change)
python ingest.py

# 2. Ask a question
python query.py "What contract vehicles does Scope use?"

# ...or run it interactively
python query.py
```

## Adding your own documents

Drop `.txt` or `.pdf` files into `data/` and re-run `python ingest.py`.

## Project layout

```
.
├── data/            # source documents (sample.txt included)
├── ingest.py        # build the vector store
├── query.py         # retrieve + answer
├── requirements.txt
└── .env.example
```
