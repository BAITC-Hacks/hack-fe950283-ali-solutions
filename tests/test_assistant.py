"""Ассистент: офлайн-шаблоны и цикл LLM-агента против фейкового OpenAI-совместимого сервера."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from moneygraph import pipeline
from moneygraph.assistant import Assistant

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ctx():
    return pipeline.compute(ROOT / "data", log=lambda *a: None)


def test_offline_common_downstream(ctx, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    G, df = ctx["G"], ctx["df"]
    cons = df[(df.role == "consolidator") & (df.in_deg >= 5)].sort_values("rank").index[0]
    payers = [str(u) for u in list(G.predecessors(cons))[:3]]
    r = Assistant(ctx).ask(f"Кто собирает деньги с {', '.join(payers)}?")
    assert r["tools"] == ["common_downstream"]
    assert str(cons) in r["answer"]


def test_offline_node_card(ctx, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    top = str(ctx["df"].sort_values("rank").index[0])
    r = Assistant(ctx).ask(f"Расскажи про {top[7:-3]}")  # короткий gid тоже понимается
    assert top in r["answer"] and r["focus"] == top


def test_llm_tool_loop(ctx, monkeypatch):
    top = str(ctx["df"].sort_values("rank").index[0])
    seen = []

    class Fake(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            if len(seen) == 1:
                msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                       "function": {"name": "node_card", "arguments": json.dumps({"gid": top})}}]}
            else:
                tool_msg = body["messages"][-1]
                assert tool_msg["role"] == "tool" and top in tool_msg["content"]
                facts=json.loads(tool_msg["content"])
                msg = {"role": "assistant", "content": json.dumps({"answer":"Проверенная карточка узла.",
                       "fact_ids":facts["evidence_refs"], "highlight_gids":[top],
                       "hypotheses":[], "limitations":[]})}
            out = json.dumps({"choices": [{"message": msg}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("AI_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    r = Assistant(ctx).ask("Кто главный?")
    srv.shutdown()
    assert r["tools"] == ["node_card"] and r["focus"] == top
    assert seen[0]["tools"] and seen[0]["messages"][0]["role"] == "system"
