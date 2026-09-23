#!/usr/bin/env python3
"""Экран просмотра + AI-ассистент на localhost (только stdlib, без внешних серверов).

    python serve.py                         # пересчёт + http://localhost:8000
    OPENAI_API_KEY=sk-... python serve.py   # ассистент на LLM (по умолчанию gpt-5.4, см. OPENAI_MODEL)
    OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=x OPENAI_MODEL=qwen2.5 python serve.py  # локальная LLM
"""
import argparse
import json
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from moneygraph import pipeline
from moneygraph.assistant import Assistant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    print("Пересчёт графа…")
    ctx = pipeline.run(Path(a.data), Path(a.out))
    bot = Assistant(ctx)
    out_dir = str(Path(a.out).resolve())

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kw):
            super().__init__(*args, directory=out_dir, **kw)

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/health":
                return self._json({"ok": True, "llm": bot.llm, "model": bot.model if bot.llm else None})
            if self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            return super().do_GET()

        def do_POST(self):
            if self.path != "/api/ask":
                return self._json({"error": "not found"}, 404)
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            q = str(req.get("question", ""))[:2000]
            hist = [m for m in req.get("history", []) if m.get("role") in ("user", "assistant")][-6:]
            return self._json(bot.ask(q, hist))

        def log_message(self, fmt, *args):
            if "/api/" in (args[0] if args else ""):
                super().log_message(fmt, *args)

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}/"
    mode = f"LLM {bot.model}" if bot.llm else "офлайн (задайте OPENAI_API_KEY для LLM)"
    print(f"\nЭкран просмотра: {url}\nАссистент: {mode}\nCtrl+C — остановить")
    if not a.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
