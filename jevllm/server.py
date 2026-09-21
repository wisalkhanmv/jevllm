"""Local web UI. Serves ui.html and streams decoding steps over SSE.

Keys can come from two places:

  * the server environment (JEV_API_KEY / ANTHROPIC_API_KEY), which is what you
    get when you run this locally with a .env file, or
  * the browser, entered by whoever is using the page — Jev key only.

Only the Jev key is ever accepted from the browser. ANTHROPIC_API_KEY, used by
the `dynamic` strategy to propose candidates, is read from the environment and
nowhere else, so a public deployment simply does not offer that strategy.

A browser-supplied key travels in a request header on a POST, never in a query
string — URLs end up in proxy logs, browser history and Referer headers, and a
credential has no business in any of them. It is held for the life of the
request and never written to disk or logged.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .api import JevClient, JevError, api_key
from .decode import Settings, Stats, decode
from .vocab import STRATEGY_HELP

UI = Path(__file__).with_name("ui.html")
ASSETS = {
    "/logo.png": ("logo.png", "image/png"),
    "/favicon.png": ("favicon.png", "image/png"),
    "/favicon.ico": ("favicon.png", "image/png"),
}

#: Ceiling on steps per request when the run is spending the *operator's* key.
#: A visitor using their own key is spending their own money, so they get the
#: higher limit. Override either with JEVLLM_MAX_STEPS / JEVLLM_MAX_STEPS_OWN.
MAX_STEPS = int(os.environ.get("JEVLLM_MAX_STEPS", "60"))
MAX_STEPS_OWN_KEY = int(os.environ.get("JEVLLM_MAX_STEPS_OWN", "400"))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # never log request lines — they can carry keys
        pass

    # -- routing ---------------------------------------------------------

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(UI.read_bytes(), "text/html; charset=utf-8")
        elif path in ASSETS:
            name, ctype = ASSETS[path]
            self._send(Path(__file__).with_name(name).read_bytes(), ctype,
                       cache="public, max-age=86400")
        elif path == "/api/config":
            # Report only whether keys exist. Never their values.
            self._send(json.dumps({
                "hasJev": bool(api_key()),
                "hasProposer": bool(os.environ.get("ANTHROPIC_API_KEY")),
                "maxSteps": MAX_STEPS,
                "maxStepsOwnKey": MAX_STEPS_OWN_KEY,
            }).encode(), "application/json")
        else:
            self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/decode":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            self.send_error(400, "bad request body")
            return
        self._decode(body)

    def _send(self, body: bytes, ctype: str, cache: str | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _event(self, payload: dict) -> None:
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        self.wfile.flush()

    def _safe_error(self, message: str) -> None:
        try:
            self._event({"error": message})
        except Exception:  # noqa: BLE001 — client already gone
            pass

    # -- keys ------------------------------------------------------------

    def _step_ceiling(self) -> int:
        """Visitors spending their own key are not rate-limited on our behalf."""
        return (MAX_STEPS_OWN_KEY if (self.headers.get("X-Jev-Key") or "").strip()
                else MAX_STEPS)

    def _keys(self) -> tuple[str | None, str | None]:
        """A browser-supplied Jev key wins; otherwise fall back to the environment.

        The proposer key is deliberately environment-only. Asking a stranger to
        paste a second credential to try a demo is a bad trade, so `dynamic`
        simply is not offered where the server has no key of its own.
        """
        header = (self.headers.get("X-Jev-Key") or "").strip()
        return (header or api_key()), os.environ.get("ANTHROPIC_API_KEY")

    # -- the stream ------------------------------------------------------

    def _decode(self, body: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        jev_key, proposer_key = self._keys()
        if not jev_key:
            self._event({"error": "No Jev API key. Enter one above, or set "
                                  "JEV_API_KEY in the server environment."})
            return

        # Mode and candidate strategy are independent: mode is what it
        # predicts, strategy is where the candidates come from.
        mode = "char" if str(body.get("mode") or "word") == "char" else "word"
        strategy = str(body.get("strategy") or "common")
        if strategy == "char":          # older clients sent it as a strategy
            mode, strategy = "char", "common"
        if strategy == "dynamic" and not proposer_key:
            self._event({"error": "The dynamic strategy is only available where the "
                                  "server has its own proposer key. Pick 'common', "
                                  "or run JevLLM locally."})
            return

        try:
            settings = Settings(
                prompt=str(body.get("prompt") or ""),
                mode=mode,
                strategy=strategy if mode == "word" else "common",
                limit=max(1, min(self._step_ceiling(), int(body.get("n", 12)))),
                temperature=float(body.get("temp", 0.2)),
                repeat_penalty=float(body.get("rp", 1.6)),
                punct_bias=float(body.get("punct", 0.6)),
                context_words=max(0, int(body.get("context", 0))),
                speculate=max(0, min(8, int(body.get("spec", 0)))),
                # Stopping is the person's call: they type what to stop at,
                # or leave it empty and stop it themselves.
                stop_at=tuple(
                    t for t in str(body.get("stopAt") or "").split("|") if t
                ),
                proposer=os.environ.get("JEVLLM_PROPOSER", "claude-sonnet-5"),
                proposer_key=proposer_key or "",
            )
        except (TypeError, ValueError):
            self._event({"error": "Bad parameter."})
            return

        stats = Stats()
        try:
            with JevClient(jev_key, timeout=settings.timeout) as client:
                for step in decode(client, settings, stats):
                    self._event({
                        "unit": step.unit,
                        "punct": step.is_punct,
                        "mode": mode,
                        "probs": dict(sorted(step.probs.items(),
                                             key=lambda kv: kv[1], reverse=True)[:9]),
                        "top": step.top_prob,
                        "confidence": step.confidence,
                        "options": step.options,
                        "steps": step.index + 1,
                        "calls": step.requests,
                        "tokens": step.tokens,
                        "cost": step.cost,
                        "elapsed": step.elapsed,
                        "speculated": step.speculated,
                    })
            self._event({"done": True, "note": STRATEGY_HELP.get(strategy, ""),
                         "acceptance": stats.acceptance,
                         "perRequest": stats.units_per_request})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except JevError as exc:
            self._safe_error(str(exc))
        except Exception as exc:  # noqa: BLE001 — surface anything to the browser
            self._safe_error(f"{type(exc).__name__}: {exc}")


def serve(port: int = 8765, host: str = "127.0.0.1") -> None:
    have = api_key()
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"JevLLM → http://localhost:{port}   (ctrl-c to stop)")
    print("key: from environment" if have else
          "key: none in environment — enter one in the page")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
