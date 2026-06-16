"""Thin JSON/HTTP wrapper over the gateway facade (spec §13 MVP "JSON/HTTP API").

Maps the §12 endpoints 1:1 onto ``Strata`` using the stdlib http.server — no external web
framework. This is a minimal, single-threaded MVP stub; auth, concurrency hardening, and
streaming are post-MVP (recorded as a deferral in docs/PINS.md). The in-process ``Strata``
facade remains the real contract.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from strata.gateway.api import Strata

# endpoint -> (facade method name, positional arg keys, keyword arg keys)
_ROUTES = {
    "/write_event": ("write_event", ["content"], []),
    "/write_memory": ("write_memory", ["content"], ["tier", "record_type", "sensitivity", "confidence"]),
    "/recall": ("recall", ["query"], ["top_k", "diversity", "budget_ms"]),
    "/update_memory": ("update_memory", ["record_id"], ["content", "status"]),
    "/supersede_memory": ("supersede_memory", ["old_id", "content"], []),
    "/delete_memory": ("delete_memory", ["record_id"], ["mode"]),
    "/deletion_status": ("deletion_status", ["job_id"], []),
    "/explain_memory": ("explain_memory", ["record_id"], []),
    "/run_reflection": ("run_reflection", ["job"], ["window"]),
}


def make_handler(strata: Strata):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence default logging
            pass

        def do_POST(self):
            route = _ROUTES.get(self.path)
            if route is None:
                return self._send(404, {"error": "unknown endpoint"})
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            method_name, pos_keys, kw_keys = route
            try:
                args = [body[k] for k in pos_keys]
                kwargs = {k: body[k] for k in kw_keys if k in body}
                result = getattr(strata, method_name)(*args, **kwargs)
                self._send(200, result)
            except KeyError as e:
                self._send(400, {"error": f"missing field {e}"})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": str(e)})

        def _send(self, code: int, payload: dict):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


class StrataHTTPServer:
    def __init__(self, strata: Strata, *, host: str = "127.0.0.1", port: int = 0) -> None:
        self.strata = strata
        self._httpd = HTTPServer((host, port), make_handler(strata))
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        return self._httpd.server_address

    def start(self) -> None:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        if self._thread:
            self._thread.join(timeout=5)
        self._httpd.server_close()
