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
from urllib.parse import urlsplit

from moneygraph import pipeline
from moneygraph.assistant import Assistant

MAX_BODY_BYTES = 16_384


def make_handler(bot, out_dir):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(out_dir), **kwargs)

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
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
                return self._json({"error": "Маршрут не найден"}, 404)
            host = self.headers.get("Host", "")
            allowed_hosts = {f"localhost:{self.server.server_port}", f"127.0.0.1:{self.server.server_port}"}
            if host not in allowed_hosts:
                return self._json({"error": "Недопустимый адрес сервера"}, 403)
            origin = self.headers.get("Origin")
            if origin and (urlsplit(origin).scheme != "http" or urlsplit(origin).netloc != host):
                return self._json({"error": "Запрос разрешён только со страницы этого приложения"}, 403)
            if self.headers.get_content_type() != "application/json":
                return self._json({"error": "Ожидается Content-Type: application/json"}, 415)
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json({"error": "Некорректный Content-Length"}, 400)
            if length <= 0:
                return self._json({"error": "Пустое тело запроса"}, 400)
            if length > MAX_BODY_BYTES:
                return self._json({"error": "Запрос слишком большой"}, 413)
            try:
                request = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError):
                return self._json({"error": "Некорректный JSON"}, 400)
            if not isinstance(request, dict):
                return self._json({"error": "Ожидается JSON-объект"}, 400)
            question = request.get("question")
            if not isinstance(question, str) or not question.strip() or len(question) > 2000:
                return self._json({"error": "Вопрос должен содержать от 1 до 2000 символов"}, 400)
            history = request.get("history", [])
            if not isinstance(history, list) or any(
                not isinstance(message, dict) or message.get("role") not in ("user", "assistant")
                or not isinstance(message.get("content"), str) or len(message["content"]) > 6000
                for message in history
            ):
                return self._json({"error": "Некорректная история диалога"}, 400)
            history = [{"role": message["role"], "content": message["content"]} for message in history[-6:]]
            try:
                response = bot.ask(question.strip(), history)
            except Exception:
                self.log_error("Ошибка обработки вопроса")
                return self._json({"error": "Не удалось обработать вопрос. Попробуйте уточнить формулировку."}, 500)
            return self._json(response)

        def log_message(self, fmt, *args):
            if "/api/" in (args[0] if args else ""):
                super().log_message(fmt, *args)

    return Handler


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

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(bot, out_dir))
    url = f"http://localhost:{a.port}/"
    mode = f"LLM {bot.model}" if bot.llm else "офлайн (задайте OPENAI_API_KEY для LLM)"
    print(f"\nЭкран просмотра: {url}\nАссистент: {mode}\nCtrl+C — остановить")
    if not a.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
