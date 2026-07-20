import os
import glob
import gradio as gr
from dotenv import load_dotenv
from langchain_groq import ChatGroq # for llm
from langchain_openai import OpenAIEmbeddings # for embeddings

load_dotenv()

# converting documents to vectors

from langchain_chroma import Chroma
from langchain_community.document_loaders   import DirectoryLoader, NotebookLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage


groq_api_key = os.environ["GROQ_API_KEY"]

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key
)

OPENAI_KEY = os.environ["OPENAI_API_KEY"]

embeddings = OpenAIEmbeddings(
    api_key=OPENAI_KEY
)

# Document location
NOTEBOOKS_DIR = "notebooks"
DB_DIR = "db"

#loading documents from the directory

notebooks = glob.glob(os.path.join(NOTEBOOKS_DIR, "**", "*.md"), recursive=True)

print(f"notebooks found: {len(notebooks)}")

loader = DirectoryLoader(NOTEBOOKS_DIR, glob="**/*.md", loader_cls=TextLoader)

documents = loader.load()


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n# ",       # h1 headers
        "\n## ",      # h2 headers
        "\n### ",     # h3 headers
        
        "\n\n",       # paragraphs
        "\n",         # lines
        " ",          # words
        "",           # characters
    ],
)

chunks = splitter.split_documents(documents)
print(f'Created {len(chunks)} chunks from {len(documents)} documents')

# Embedding and storing the chunks in ChromaDB

if os.path.exists(DB_DIR):
    print(f"Loading existing ChromaDB from {DB_DIR}")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    vector_count = vectorstore._collection.count()
    print(f"Vectorstore created with {vector_count} vectors")


    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    results = retriever.invoke("What are the features of CarUp?")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---\n{doc.page_content}")
