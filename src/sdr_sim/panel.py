"""Painel web do simulador: uma página e três rotas JSON.

Biblioteca padrão só (`http.server`), de propósito: o simulador tem duas
dependências e continua com duas. Um framework web para servir uma página e
três rotas seria mais peso do que o próprio simulador.

    GET  /               a página
    GET  /api/state      sintonia, SNR, emissores, Doppler, pacotes
    GET  /api/spectrum   espectro do último bloco, em dB
    POST /api/control    muda o simulador (ver SimController.apply)

Sem autenticação: é ferramenta de bancada. O compose publica a porta só em
127.0.0.1 — quem alcança o painel já está na máquina.
"""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

from sdr_sim.control import SimController

MAX_BODY_BYTES = 16 * 1024


def _page() -> bytes:
    return resources.files("sdr_sim").joinpath("panel.html").read_bytes()


def make_handler(controller: SimController) -> type[BaseHTTPRequestHandler]:
    page = _page()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            # O painel consulta o estado várias vezes por segundo; logar cada
            # GET afogaria o log do simulador, que é onde está o que importa.
            pass

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, payload) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(HTTPStatus.OK, controller.state())
            elif self.path == "/api/spectrum":
                self._json(HTTPStatus.OK, controller.spectrum_db())
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "rota desconhecida"})

        def do_POST(self) -> None:
            if self.path != "/api/control":
                self._json(HTTPStatus.NOT_FOUND, {"error": "rota desconhecida"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if not 0 < length <= MAX_BODY_BYTES:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "corpo ausente ou grande demais"})
                return

            try:
                changes = json.loads(self.rfile.read(length))
                controller.apply(changes)
            except (ValueError, UnicodeDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return

            print(f"[sdr-sim] painel: {json.dumps(changes, ensure_ascii=False)}", flush=True)
            self._json(HTTPStatus.OK, controller.state())

    return Handler


def start_panel(controller: SimController, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(controller))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="panel").start()

    return server
