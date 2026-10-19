"""O OBDH do FS-2 recebendo telecomandos: o `process_tc.c`, escrito de novo aqui.

Fonte: `spacelab-ufsc/obdh2`, `firmware/app/tasks/process_tc.c`, tag v1.0,
commit 3f0ad36. Cada comportamento abaixo cita o handler de lá. Os pares
TC -> resposta estão em `uplink-pares-e-estados-fs2.md` §2.

Escrito de novo, sem importar o `grs-uplink` nem o `grs-telemetry-decoder`: o
satélite simulado é o juiz dos dois. Se o uplink montar o HMAC errado, é aqui
que o TC morre calado — como morreria no satélite.

O que o modelo reproduz (v1.0):

- **Ping (0x40):** `pkt_len >= 8`, sem HMAC. Responde 0x02 pelo TTC_1 (VHF)
  com os 7 bytes de quem pediu (`process_tc_ping_request`).
- **Activate/Deactivate Module (0x45/0x46):** `pkt_len >= 29`, HMAC-SHA1 sobre
  os 9 primeiros bytes, com a chave do TC. Autenticado, atualiza o "último TC
  válido". Módulo 1, 2 ou 3: responde 0x13 pelo TTC_0 (UHF). Outro módulo: o
  `default: break` — silêncio, mas o TC conta como válido.
- **HMAC inválido:** silêncio. Só o log de bordo registraria.
- **ID desconhecido:** silêncio ("Invalid TC request!").

O que ele NÃO reproduz: hibernação, a fila do Schedule TC, os outros
telecomandos, o buffer de 5 posições do TTC (o TC chega um por vez aqui).

Para testar o lado da estação, dá para mandar o satélite falhar o próximo TC
com um código de erro, ou ignorá-lo (`fail_next`, `drop_next`).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import struct
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SATELLITE_CALLSIGN = b" PY0EFS"      # CONFIG_SATELLITE_CALLSIGN, 7 bytes
CALLSIGN_LENGTH = 7
HMAC_LENGTH = 20
KEY_LENGTH = 16

PING_REQUEST = 0x40
ACTIVATE_MODULE = 0x45
DEACTIVATE_MODULE = 0x46
PING_ANSWER = 0x02
TC_FEEDBACK = 0x13

MODULE_IDS = {1, 2, 3}               # module_id_e do satellite.h
SUCCESS = 0                          # ERRNO_FB_SUCESSFULL_EXEC

# Que chave cada TC privado usa (CONFIG_TC_KEY_* do keys.h). O nome é o do
# arquivo de chaves, o mesmo formato do grs-uplink.
KEY_NAMES = {ACTIVATE_MODULE: "activate_module", DEACTIVATE_MODULE: "deactivate_module"}

RECENT = 20


@dataclass(frozen=True)
class Response:
    radio: str          # "vhf" (TTC_1) ou "uhf" (TTC_0)
    packet: bytes       # o pacote FSat, sem NGHam


def load_keys(path: str | None) -> dict[str, bytes]:
    """O mesmo JSON do grs-uplink: {nome: 16 caracteres ASCII}; chave com `_` é comentário."""
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = {}
    for name, value in raw.items():
        if name.startswith("_"):
            continue
        if not isinstance(value, str) or len(value.encode("ascii")) != KEY_LENGTH:
            raise ValueError(f"chave {name!r}: precisa ter {KEY_LENGTH} caracteres ASCII")
        keys[name] = value.encode("ascii")
    return keys


class SatelliteModel:
    def __init__(self, keys: Mapping[str, bytes], clock: Callable[[], float] = time.time) -> None:
        self._keys = dict(keys)
        self._clock = clock
        self._lock = threading.Lock()
        self.last_valid_tc = 0
        self.ts_last_contact = 0
        self.received = 0
        self.rejected = 0
        self._fail_next: int | None = None
        self._drop_next = False
        self._recent: deque[dict[str, Any]] = deque(maxlen=RECENT)

    # --- injeção de falha (painel) ---------------------------------------------

    def fail_next(self, error_code: int) -> None:
        """O próximo TC com feedback responde com este código (ex.: -2)."""
        if not -32768 <= error_code <= 32767:
            raise ValueError("o código de erro é int16")
        with self._lock:
            self._fail_next = error_code

    def drop_next(self) -> None:
        """O próximo TC é ignorado, como se não tivesse chegado."""
        with self._lock:
            self._drop_next = True

    # --- process_tc ----------------------------------------------------------------

    def process(self, packet: bytes) -> list[Response]:
        with self._lock:
            self.received += 1
            if self._drop_next:
                self._drop_next = False
                return self._log(packet, "ignorado (injeção de falha)", [])

            if len(packet) < 1 + CALLSIGN_LENGTH:
                return self._log(packet, "curto demais", [])

            tc_id = packet[0]
            if tc_id == PING_REQUEST:
                return self._ping(packet)
            if tc_id in (ACTIVATE_MODULE, DEACTIVATE_MODULE):
                return self._module(packet)
            return self._log(packet, "TC desconhecido (Invalid TC request!)", [])

    def _ping(self, packet: bytes) -> list[Response]:
        self.last_valid_tc, self.ts_last_contact = packet[0], int(self._clock())
        pong = bytes([PING_ANSWER]) + SATELLITE_CALLSIGN + packet[1:1 + CALLSIGN_LENGTH]
        return self._log(packet, "pong", [Response("vhf", pong)])

    def _module(self, packet: bytes) -> list[Response]:
        body_length = 1 + CALLSIGN_LENGTH + 1
        if len(packet) < body_length + HMAC_LENGTH:
            return self._log(packet, "curto demais para o HMAC", [])

        key = self._keys.get(KEY_NAMES[packet[0]])
        tag = packet[body_length:body_length + HMAC_LENGTH]
        if key is None or not hmac.compare_digest(
                hmac.new(key, packet[:body_length], hashlib.sha1).digest(), tag):
            self.rejected += 1
            return self._log(packet, "HMAC inválido (Invalid key!)", [])

        self.last_valid_tc, self.ts_last_contact = packet[0], int(self._clock())
        if packet[8] not in MODULE_IDS:
            return self._log(packet, f"módulo {packet[8]} desconhecido (default: break)", [])

        code = SUCCESS
        if self._fail_next is not None:
            code, self._fail_next = self._fail_next, None
        return self._log(packet, f"feedback {code}", [Response("uhf", self._feedback(packet, code))])

    def _feedback(self, packet: bytes, error_code: int) -> bytes:
        """send_tc_feedback: quem pediu (7), ID do TC (1), relógio (4), erro int16 (2)."""
        return (bytes([TC_FEEDBACK]) + SATELLITE_CALLSIGN + packet[1:1 + CALLSIGN_LENGTH]
                + struct.pack(">BIh", packet[0], int(self._clock()), error_code))

    def _log(self, packet: bytes, outcome: str, responses: list[Response]) -> list[Response]:
        self._recent.appendleft({
            "at": round(self._clock(), 1),
            "tc_id": f"0x{packet[0]:02X}" if packet else None,
            "requester": packet[1:1 + CALLSIGN_LENGTH].decode("latin-1").strip() if len(packet) > 1 else "",
            "length": len(packet),
            "outcome": outcome,
            "responses": [{"radio": r.radio, "pkt_id": f"0x{r.packet[0]:02X}"} for r in responses],
        })
        return responses

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "received": self.received,
                "rejected_hmac": self.rejected,
                "last_valid_tc": f"0x{self.last_valid_tc:02X}" if self.last_valid_tc else None,
                "ts_last_contact": self.ts_last_contact or None,
                "fail_next": self._fail_next,
                "drop_next": self._drop_next,
                "recent": list(self._recent),
            }
