"""Ingest documents into a local Chroma vector store.

Reads every file under ./data, splits it into overlapping chunks, embeds the
chunks with a local sentence-transformers model, and persists them to a Chroma
database in ./chroma_db.

Run this once (or whenever your documents change) before querying:

    python ingest.py
"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(data_dir: str):
    """Load .txt and .pdf files from data_dir into LangChain documents."""
    loaders = [
        DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs


def main():
    if not Path(DATA_DIR).exists():
        raise SystemExit(f"No '{DATA_DIR}/' directory found. Add documents first.")

    docs = load_documents(DATA_DIR)
    if not docs:
        raise SystemExit(f"No .txt or .pdf files found in '{DATA_DIR}/'.")
    print(f"Loaded {len(docs)} document(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunk(s).")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"Persisted vector store to '{PERSIST_DIR}/'.")


if __name__ == "__main__":
    main()
