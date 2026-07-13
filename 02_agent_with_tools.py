"""
LangChain - Agent with Tools
=============================
Demonstrates an AI agent that can use tools (calculator, web search placeholder).
Run with:  python 02_agent_with_tools.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM (Ollama Cloud - FREE) ────────────────────────────────────
# Make sure your chosen provider has tool/function calling support.
# Cloud models (free tier): qwen3-coder-next:cloud, nemotron-3-super:cloud, minimax-m2.5:cloud

from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen3-coder-next:cloud", temperature=0)

# ── Tools ───────────────────────────────────────────────────────
from langchain_core.tools import tool


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Hasil: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return f"Sekarang: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


tools = [calculate, get_current_time]

# ── Agent ───────────────────────────────────────────────────────
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

agent = create_react_agent(llm, tools)

query = "Jam berapa sekarang? Hitung juga 2 + 3 * 4 berapa?"

print(f"\n>>> User: {query}")
print("=" * 60)

for chunk in agent.stream({"messages": [HumanMessage(content=query)]}):
    for node_name, node_data in chunk.items():
        if "messages" in node_data:
            for msg in node_data["messages"]:
                if hasattr(msg, "content") and msg.content:
                    role = msg.type.upper() if hasattr(msg, "type") else node_name.upper()
                    print(f"\n[{role}]: {msg.content}")
