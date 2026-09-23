#!/usr/bin/env python3
"""Local viewer, read-only analysis queries and optional AI."""
import argparse
import json
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from moneygraph import pipeline
from moneygraph.assistant import Assistant
from moneygraph.analysis_service import AnalysisService
from moneygraph.contracts import clean


def make_handler(ctx, out_dir):
    service=AnalysisService(ctx)
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kw):
            super().__init__(*args,directory=str(out_dir),**kw)

        def _json(self,obj,code=200):
            body=json.dumps(clean(obj),ensure_ascii=False,allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type","application/json; charset=utf-8")
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Cache-Control","no-store")
            self.end_headers()
            self.wfile.write(body)

        def _current(self):
            try:
                disk=json.loads((Path(out_dir)/"run_manifest.json").read_text(encoding="utf-8"))
                return disk["run_id"]==ctx["manifest"]["run_id"]
            except (OSError,ValueError,KeyError):
                return False

        def _download(self,name,body,kind):
            self.send_response(200)
            self.send_header("Content-Type",kind+"; charset=utf-8")
            self.send_header("Content-Disposition",'attachment; filename="'+name+'"')
            self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith(("/download/","/api/")) and not self._current():
                return self._json({"error":"Расчёт изменился: перезапустите serve.py и обновите страницу"},409)
            if self.path.startswith("/download/"):
                name=self.path.removeprefix("/download/")
                if name not in ("nodes_roles.csv","clusters.csv","top_nodes.csv"):
                    return self._json({"error":"not found"},404)
                return self._download(name,(Path(out_dir)/name).read_bytes(),"text/csv")
            if self.path.startswith("/api/node-report/") and self.path.endswith(".md"):
                gid=self.path.removeprefix("/api/node-report/")[:-3]
                try:
                    report=service.get_node_report(gid)["result"]["markdown"]
                    return self._download("node-"+gid+".md",report.encode(),"text/markdown")
                except ValueError as exc:
                    return self._json({"error":str(exc)},400)

            if self.path=="/api/health":
                bot=Assistant(ctx)
                return self._json({"ok":True,"llm":bot.llm,"model":bot.model if bot.llm else None,
                                   "run_id":ctx["manifest"]["run_id"]})
            if self.path=="/favicon.ico":
                self.send_response(204); self.end_headers(); return
            return super().do_GET()

        def do_POST(self):
            if not self._current():
                return self._json({"error":"Расчёт изменился: перезапустите serve.py и обновите страницу"},409)
            if self.path not in ("/api/ask","/api/query"):
                return self._json({"error":"not found"},404)
            # Browser calls must be same-origin; localhost tools remain usable without Origin.
            origin=self.headers.get("Origin")
            if origin and origin != "http://"+self.headers.get("Host",""):
                return self._json({"error":"Origin не совпадает"},403)
            try:
                n=int(self.headers.get("Content-Length","0"))
                if not 0<n<=65536: return self._json({"error":"Размер запроса 1..65536 байт"},413)
                req=json.loads(self.rfile.read(n))
                if not isinstance(req,dict): raise ValueError("Нужен JSON-объект")
                if req.get("run_id",ctx["manifest"]["run_id"])!=ctx["manifest"]["run_id"]:
                    return self._json({"error":"Run изменился; обновите страницу"},409)
                if self.path=="/api/query":
                    return self._json(service.query(req.get("action"),req.get("args",{})))
                question=req.get("question","")
                history=req.get("history",[])
                if not isinstance(question,str) or not question.strip() or len(question)>2000:
                    raise ValueError("Вопрос: 1..2000 символов")
                if not isinstance(history,list): raise ValueError("history должен быть списком")
                return self._json(Assistant(ctx).ask(question,history[-6:]))
            except (ValueError,TypeError,KeyError) as exc:
                return self._json({"error":str(exc)},400)

        def log_message(self,*args):
            pass
    return Handler


def main():
    if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data")
    ap.add_argument("--out",default="out")
    ap.add_argument("--port",type=int,default=8000)
    ap.add_argument("--period-end")
    ap.add_argument("--no-open",action="store_true")
    a=ap.parse_args()
    ctx=pipeline.run(Path(a.data),Path(a.out),period_end=a.period_end)
    srv=ThreadingHTTPServer(("127.0.0.1",a.port),make_handler(ctx,Path(a.out).resolve()))
    url=f"http://localhost:{srv.server_port}/"
    print(f"Экран и локальная аналитика: {url}; run {ctx['manifest']['run_id']}")
    if not a.no_open: threading.Timer(0.8,lambda:webbrowser.open(url)).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
    finally: srv.server_close()

if __name__=="__main__":
    main()
