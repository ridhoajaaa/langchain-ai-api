"""
LangChain API Test Client
==========================
Test all endpoints of the LangChain FastAPI server.
Run:  python test_client.py
"""

import json
import httpx
import sys

BASE_URL = "http://localhost:8000"


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_response(label, data):
    print(f"\n  {label}:")
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str) and len(v) > 150:
                print(f"    {k}: {v[:150]}...")
            else:
                print(f"    {k}: {v}")
    else:
        print(f"    {data}")


async def test_all():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60) as client:
        # ── 1. Health ──────────────────────────────────────
        print_header("1. GET /health")
        r = await client.get("/health")
        print_response("Status", r.json())
        assert r.status_code == 200, f"Health failed: {r.status_code}"
        print("  ✅ OK")

        # ── 2. Root ────────────────────────────────────────
        print_header("2. GET /")
        r = await client.get("/")
        data = r.json()
        print_response("Endpoints", list(data.get("endpoints", {}).keys()))
        assert r.status_code == 200
        print("  ✅ OK")

        # ── 3. Chat ────────────────────────────────────────
        print_header("3. POST /chat")
        r = await client.post("/chat", json={
            "message": "Apa itu LangChain? Jawab singkat.",
            "provider": "ollama",
            "temperature": 0.7,
        })
        data = r.json()
        print_response("Response", data)
        assert r.status_code == 200, f"Chat failed: {r.status_code}"
        assert "response" in data
        print("  ✅ OK")

        # ── 4. Agent ───────────────────────────────────────
        print_header("4. POST /agent")
        r = await client.post("/agent", json={
            "message": "Hitung 15 + 27 dan jam berapa sekarang?",
            "provider": "ollama",
        })
        data = r.json()
        print_response("Response", data)
        assert r.status_code == 200, f"Agent failed: {r.status_code}"
        assert "response" in data
        print("  ✅ OK")

        # ── 5. RAG Query ───────────────────────────────────
        print_header("5. POST /rag/query")
        r = await client.post("/rag/query", json={
            "query": "Apa itu RAG?",
            "provider": "ollama",
        })
        data = r.json()
        print_response("Answer", data)
        assert r.status_code == 200, f"RAG query failed: {r.status_code}"
        assert "answer" in data
        print("  ✅ OK")

        # ── 6. RAG Ingest ──────────────────────────────────
        print_header("6. POST /rag/ingest")
        r = await client.post("/rag/ingest", json={
            "provider": "ollama",
            "documents": [
                {
                    "page_content": "Python adalah bahasa pemrograman interpretatif yang "
                                    "populer untuk data science, web development, dan AI.",
                    "metadata": {"source": "custom", "topic": "python"},
                }
            ],
        })
        data = r.json()
        print_response("Ingest result", data)
        assert r.status_code == 200, f"Ingest failed: {r.status_code}"
        assert data["status"] == "ok"
        print("  ✅ OK")

        # ── 7. SSE Stream ──────────────────────────────────
        print_header("7. GET /agent/stream (SSE)")
        async with client.stream("GET", "/agent/stream",
                                  params={"message": "Halo!", "provider": "ollama"}) as resp:
            assert resp.status_code == 200, f"Stream failed: {resp.status_code}"
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    data = json.loads(line[5:])
                    if data.get("status") == "completed":
                        break
        print("  ✅ OK (SSE streaming completed)")

        # ── Summary ────────────────────────────────────────
        print_header("✅ ALL TESTS PASSED!")
        print("  Server is running correctly.")
        print("  Open http://localhost:8000/docs for interactive API docs.")


if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(test_all())
    except httpx.ConnectError:
        print("\n❌ Cannot connect to server.")
        print("   Start the server first:")
        print("   cd ~/DataD/Langchain")
        print("   source venv/bin/activate")
        print("   uvicorn server:app --reload --host 0.0.0.0 --port 8000")
        sys.exit(1)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
