# RAG Assistant (LangChain + Groq + Chroma)

A Retrieval-Augmented Generation (RAG) assistant that indexes local markdown
notebooks into a vector store and answers questions using only the retrieved
content.

## Stack

| Role | Component |
| --- | --- |
| Orchestration | LangChain |
| LLM (generation) | Groq — `llama-3.1-8b-instant` |
| Embeddings | OpenAI (`OpenAIEmbeddings`) |
| Vector store | Chroma (persisted locally to `db/`) |
| UI (planned) | Gradio |

You need two API keys: one for **Groq** (generation) and one for **OpenAI**
(embeddings).

## How it works

`RAG assistant.py` is the main script. It:

1. Loads every `.md` file under `notebooks/`.
2. Splits them into overlapping chunks (1000 chars, 200 overlap), preferring
   markdown header boundaries.
3. Embeds the chunks with OpenAI embeddings and stores them in a local Chroma
   database (`db/`).
4. Retrieves the most relevant chunks for a question and passes them to the
   Groq LLM to answer.

> **Status:** work in progress. The Gradio dependency is included for an
> upcoming chat UI, but the current script runs an indexing + sample-retrieval
> pass from the command line.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then add your GROQ_API_KEY and OPENAI_API_KEY
```

`.env` is git-ignored, so your keys are never committed.

## Usage

```bash
python "RAG assistant.py"
```

## Adding your own documents

Drop `.md` files into `notebooks/` and re-run the script to rebuild the index.

## Project layout

```
.
├── RAG assistant.py   # main app: ingest notebooks + retrieve + answer
├── notebooks/         # source markdown documents to index
├── data/              # sample data
├── db/                # persisted Chroma vector store (git-ignored, generated)
├── requirements.txt
├── .env.example       # template for GROQ_API_KEY and OPENAI_API_KEY
├── ingest.py          # legacy scaffold (HuggingFace embeddings) — not current
└── query.py           # legacy scaffold (Anthropic) — not current
```

## Notes

- `ingest.py` and `query.py` are earlier scaffolding from an initial
  Claude + local-embeddings prototype. They are kept for reference but are not
  the current pipeline and their dependencies are not all in
  `requirements.txt`.
