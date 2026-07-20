"""Ask questions against the ingested documents.

Retrieves the most relevant chunks from the Chroma store and asks Claude to
answer using only that context.

Usage:
    python query.py "What contract vehicles does Scope use?"
    python query.py            # starts an interactive prompt
"""

import sys

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Anthropic's most capable model as of this project.
CHAT_MODEL = "claude-opus-4-8"

PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the question using only the context "
    "below. If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 4})
    llm = ChatAnthropic(model=CHAT_MODEL, temperature=0)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


def main():
    load_dotenv()
    chain = build_chain()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(chain.invoke(question))
        return

    print("Ask a question (Ctrl-C to quit).")
    try:
        while True:
            question = input("\n> ").strip()
            if question:
                print("\n" + chain.invoke(question))
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()
