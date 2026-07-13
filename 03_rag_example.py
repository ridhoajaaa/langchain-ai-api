"""
LangChain - RAG (Retrieval Augmented Generation) Example
=========================================================
Demonstrates: document loading -> chunking -> embedding -> vector store -> retrieval -> QA
Run with:  python 03_rag_example.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── 1. Load a document ─────────────────────────────────────────
def format_docs(docs):
    """Format retrieved documents into context string."""
    return "\n\n".join(d.page_content for d in docs)

from langchain_core.documents import Document

# Sample documents (in real use, load from PDF, web, etc.)
documents = [
    Document(
        page_content="LangChain adalah framework untuk membangun aplikasi berbasis LLM. "
                     "Mendukung rantai (chains), agen (agents), dan RAG.",
        metadata={"source": "docs", "topic": "introduction"},
    ),
    Document(
        page_content="RAG (Retrieval Augmented Generation) menggabungkan retrieval informasi "
                     "eksternal dengan generative LLM untuk menghasilkan jawaban yang akurat.",
        metadata={"source": "docs", "topic": "rag"},
    ),
    Document(
        page_content="LangGraph adalah library untuk membangun agent multi-langkah "
                     "dengan state graph yang dapat bercabang dan berulang.",
        metadata={"source": "docs", "topic": "langgraph"},
    ),
]

# ── 2. Split into chunks & embed ───────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Use free local embeddings (no API key needed)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

# ── 3. Store in vector DB ──────────────────────────────────────
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ── 4. QA Chain (Ollama Cloud - FREE) ──────────────────────────
# Cloud models (free tier): qwen3-coder-next:cloud, nemotron-3-super:cloud
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="qwen3-coder-next:cloud", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Jawab pertanyaan berdasarkan konteks berikut:\n\n{context}"),
    ("human", "{input}"),
])

from langchain_core.runnables import RunnablePassthrough
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
)

# ── 5. Ask questions ──────────────────────────────────────────
questions = [
    "Apa itu RAG?",
    "Apa perbedaan LangChain dan LangGraph?",
]

for q in questions:
    result = rag_chain.invoke(q)
    print(f"\n>>> User: {q}")
    print(f"[AI]: {result.content if hasattr(result, 'content') else result}")
    print("-" * 60)
