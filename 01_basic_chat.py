"""
LangChain - Basic Chat Example
===============================
Run with:  python 01_basic_chat.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Pick your provider ──────────────────────────────────────────
# Uncomment ONE of the following providers:

# Option 1: Ollama Cloud (FREE, no download / no API key needed)
#   Cloud models: qwen3-coder-next:cloud, nemotron-3-super:cloud, dll.
#   Install Ollama: https://ollama.com  (already installed ✅)
from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen3-coder-next:cloud", temperature=0.7)

# Option 2: OpenAI
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Option 3: Anthropic Claude
# from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.7)

# Option 4: Google Gemini
# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# ── Chat ────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="Kamu adalah asisten AI yang ramah dan membantu. Jawab dalam Bahasa Indonesia."),
    HumanMessage(content="Apa itu LangChain? Jelaskan secara singkat."),
]

response = llm.invoke(messages)
print("\n" + "=" * 60)
print("JAWABAN:")
print("=" * 60)
print(response.content)
