"""Jev client and sampling. No third-party dependencies.

The client holds one TLS connection open for the life of a run. Each decoding
step is a separate HTTP request, and on this API a fresh TCP + TLS handshake
costs ~610ms against ~390ms of actual model time — so reconnecting per step
made two thirds of the wall clock pure setup.
"""

from __future__ import annotations

import http.client
import json
import os
import random
import ssl
import time
import urllib.parse

API_URL = "https://api.typesafe.ai/v1/systemone"

#: Jev bills input at $42 per billion tokens. Output tokens are free.
USD_PER_INPUT_TOKEN = 42.0 / 1_000_000_000

#: Most options allowed in a single Choice. Byte-level decoding needs 256,
#: which is exactly one too many.
MAX_OPTIONS = 255


class JevError(RuntimeError):
    pass


def api_key() -> str | None:
    return os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")


class JevClient:
    """A persistent-connection Jev client.

    Usable as a context manager. Reconnects transparently if the server drops
    the connection between steps.
    """

    def __init__(self, key: str, url: str = API_URL, timeout: float = 45.0) -> None:
        parts = urllib.parse.urlparse(url)
        self.host = parts.netloc
        self.path = parts.path or "/v1/systemone"
        self.key = key
        self.timeout = timeout
        self._conn: http.client.HTTPSConnection | None = None
        self.requests = 0
        self.input_tokens = 0
        self.reconnects = 0

    # -- connection ------------------------------------------------------

    def _connect(self) -> http.client.HTTPSConnection:
        if self._conn is None:
            self._conn = http.client.HTTPSConnection(
                self.host, timeout=self.timeout, context=ssl.create_default_context()
            )
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def __enter__(self) -> "JevClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- requests --------------------------------------------------------

    def _post(self, body: str) -> tuple[int, str, dict]:
        conn = self._connect()
        conn.request("POST", self.path, body=body, headers={
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        })
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8", "replace"), dict(resp.getheaders())

    def ask(self, state, questions: dict, *, model: str = "jev-latest",
            attempts: int = 5) -> dict:
        """Send a batch of questions against one state.

        Jev reads the state once and evaluates every question against it in
        parallel, so N questions in one request cost roughly one state's worth
        of tokens rather than N.
        """
        body = json.dumps({"state": state, "model": model, "questions": questions})
        delay = 1.0

        for attempt in range(attempts):
            try:
                status, text, headers = self._post(body)
            except (http.client.HTTPException, OSError) as exc:
                # Stale keep-alive connection, or a genuine network fault.
                self.close()
                self.reconnects += 1
                if attempt == attempts - 1:
                    raise JevError(f"Could not reach Jev: {exc}") from exc
                time.sleep(delay)
                delay *= 2
                continue

            if status == 200:
                data = json.loads(text)
                self.requests += 1
                self.input_tokens += data.get("usage", {}).get("input_tokens", 0)
                return data
            if status == 429:
                self.close()
                wait = float(headers.get("retry-after") or delay)
                if attempt == attempts - 1:
                    raise JevError("Rate limited, and out of retries.")
                time.sleep(max(wait, delay))
                delay *= 2
                continue
            self.close()
            if status in (401, 403):
                raise JevError("Jev rejected the API key (check JEV_API_KEY).")
            raise JevError(f"Jev returned {status}: {text[:300]}")

        raise JevError("unreachable")

    @property
    def cost(self) -> float:
        return self.input_tokens * USD_PER_INPUT_TOKEN


def choice_question(instructions: str, options) -> dict:
    """Build a Choice question. `options` may be a list or an option->hint map."""
    criteria = options if isinstance(options, dict) else {o: None for o in options}
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def read_choice(response: dict, qid: str) -> tuple[dict[str, float], float | None]:
    """Pull (probabilities, confidence) for one question id out of a response."""
    answers = response.get("answers") or {}
    answer = answers.get(qid)
    if answer is None:
        raise JevError(f"Jev returned no answer for {qid!r}.")
    probs = answer.get("probabilities") or {answer["choice"]: 1.0}
    return probs, answer.get("confidence")


def sample(probs: dict[str, float], *, temperature: float = 0.7, top_k: int = 0,
           top_p: float = 1.0, rng: random.Random | None = None) -> str:
    """Sample one option. temperature <= 0 is greedy.

    Jev returns probabilities rather than logits, so temperature is applied as
    p**(1/T) and renormalised — equivalent to scaling the underlying logits.
    """
    rng = rng or random.Random()
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        raise JevError("Empty distribution.")
    if temperature <= 0:
        return ranked[0][0]
    if top_k > 0:
        ranked = ranked[:top_k]
    if 0 < top_p < 1.0:
        kept, running = [], 0.0
        for option, p in ranked:
            kept.append((option, p))
            running += p
            if running >= top_p:
                break
        ranked = kept
    weights = [p ** (1.0 / temperature) if p > 0 else 0.0 for _, p in ranked]
    if sum(weights) <= 0:
        return ranked[0][0]
    return rng.choices([o for o, _ in ranked], weights=weights, k=1)[0]
