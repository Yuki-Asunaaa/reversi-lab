#!/usr/bin/env python3
"""启动本地黑白棋实验室；只需 Python 3.9+，不需要安装第三方依赖。"""

import argparse
import csv
from http.cookies import SimpleCookie
import io
import json
import mimetypes
import os
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from reversi.analysis import MODEL_PATH
from reversi.session import Session

ROOT = Path(__file__).resolve().parent


class SessionRegistry:
    """公开部署时按浏览器 Cookie 隔离棋局，并清理长期空闲会话。"""

    def __init__(self, ttl_seconds=1800, max_sessions=64):
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.lock = threading.Lock()
        self.sessions = {}

    def get(self, session_id):
        now = time.monotonic()
        expired = []
        with self.lock:
            for key, (last_seen, session) in list(self.sessions.items()):
                if now - last_seen > self.ttl_seconds:
                    expired.append(session)
                    del self.sessions[key]
            if session_id in self.sessions:
                self.sessions[session_id] = (now, self.sessions[session_id][1])
                current_id = session_id
                current = self.sessions[current_id][1]
            elif len(self.sessions) < self.max_sessions:
                current_id = secrets.token_urlsafe(24)
                current = Session(persist_experiment=False)
                self.sessions[current_id] = (now, current)
            else:
                current_id = current = None
        for session in expired:
            session.close()
        return current_id, current

    def close(self):
        with self.lock:
            sessions = [session for _, session in self.sessions.values()]
            self.sessions.clear()
        for session in sessions:
            session.close()


class SharedSessionRegistry:
    """本地单人运行时沿用一盘共享棋局，不发送会话 Cookie。"""

    def __init__(self, session):
        self.session = session

    def get(self, session_id):
        return None, self.session

    def close(self):
        self.session.close()


def make_handler(session_registry):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def prepare_session(self):
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            morsel = cookie.get("reversi_session")
            session_id = morsel.value if morsel else None
            self.session_id, self.session = session_registry.get(session_id)
            self.set_session_cookie = bool(self.session and self.session_id != session_id)
            if self.session is None:
                self.respond({"error": "当前访问人数较多，请稍后重试"}, 503)
                return False
            return True

        def respond(
            self, data, status=200, content_type="application/json; charset=utf-8", filename=None
        ):
            if isinstance(data, (dict, list)):
                data = json.dumps(data, ensure_ascii=False).encode("utf-8")
            elif isinstance(data, str):
                data = data.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if getattr(self, "set_session_cookie", False):
                cookie = "reversi_session=%s; Path=/; HttpOnly; SameSite=Lax" % self.session_id
                if os.environ.get("RENDER"):
                    cookie += "; Secure"
                self.send_header("Set-Cookie", cookie)
                self.set_session_cookie = False
            if filename:
                self.send_header("Content-Disposition", 'attachment; filename="%s"' % filename)
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self.prepare_session():
                return
            path = urlparse(self.path).path
            if path == "/api/state":
                return self.respond(self.session.state())
            if path == "/api/evolution":
                evidence = ROOT / "results" / "evolution" / "summary.json"
                return self.respond(
                    json.loads(evidence.read_text(encoding="utf-8")) if evidence.exists() else {}
                )
            if path == "/api/calibration":
                model = (
                    json.loads(MODEL_PATH.read_text(encoding="utf-8"))
                    if MODEL_PATH.exists()
                    else {"status": "missing"}
                )
                return self.respond(model)
            if path == "/api/export/game":
                return self.respond(self.session.export_game(), filename="reversi-game.json")
            if path == "/api/export/experiment":
                with self.session.lock:
                    return self.respond(self.session.experiment, filename="reversi-experiment.json")
            if path == "/api/export/csv":
                output = io.StringIO()
                with self.session.lock:
                    rows = self.session.experiment.get("summary", [])
                    if rows:
                        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                return self.respond(
                    "\ufeff" + output.getvalue(),
                    content_type="text/csv; charset=utf-8",
                    filename="reversi-results.csv",
                )
            target = (ROOT / "web" / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
            if ROOT / "web" not in target.parents or not target.is_file():
                return self.respond({"error": "未找到页面"}, 404)
            return self.respond(
                target.read_bytes(),
                content_type=mimetypes.guess_type(str(target))[0] or "application/octet-stream",
            )

        def do_POST(self):
            if not self.prepare_session():
                return
            # 仅同源页面可修改对局，兼容公开 HTTPS 反向代理。
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc.lower() != self.headers.get("Host", "").lower():
                return self.respond({"error": "不允许跨站请求"}, 403)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size > 100000:
                    raise ValueError("请求过大")
                payload = json.loads(self.rfile.read(size) or b"{}")
                action = urlparse(self.path).path.removeprefix("/api/")
                return self.respond(self.session.command(action, payload))
            except (ValueError, KeyError, TypeError) as error:
                return self.respond({"error": str(error)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    public = bool(os.environ.get("PORT") or os.environ.get("RENDER"))
    application = SessionRegistry() if public else SharedSessionRegistry(Session())
    try:
        host = "0.0.0.0" if public else "127.0.0.1"
        server = ThreadingHTTPServer((host, args.port), make_handler(application))
    except OSError as error:
        application.close()
        parser.exit(1, "端口无法使用：%s。可使用 --port 8766 指定其它端口。\n" % error)
    url = "http://127.0.0.1:%s" % server.server_port
    if public:
        print("Reversi Lab 已启动，监听端口 %s。" % server.server_port, flush=True)
        print("按 Ctrl+C 结束服务。", flush=True)
    else:
        print("Reversi Lab 已启动：" + url, flush=True)
        print("按 Ctrl+C 结束服务。所有算法与数据均在本机运行。", flush=True)
    if not public and not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        application.close()
        server.server_close()


if __name__ == "__main__":
    main()
