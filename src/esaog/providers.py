"""Real LLM providers behind one `complete(system, user) -> Completion` interface.

  anthropic : official Anthropic Python SDK (credentials from ANTHROPIC_API_KEY or an `ant auth` profile)
  openai    : official OpenAI Python SDK (OPENAI_API_KEY)
  ollama    : local Ollama HTTP API (http://localhost:11434)

All providers run at temperature 0 where the model allows it. Responses are cached on disk keyed by
(provider, model, system, user): an identical prompt issued by two methods receives the identical
completion (the real-LLM analogue of common random numbers). The cache policy is reported in the manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "results" / "llm_cache"

# USD per token (input, output). Anthropic from the published price list; others must be supplied.
PRICES = {
    ("openai", "gpt-6-sol"): (2.00e-6, 10.00e-6), ("openai", "gpt-6-luna"): (0.10e-6, 0.50e-6),
    ("openai", "gpt-6-astra"): (10.0e-6, 50.0e-6), ("openai", "gpt-5.5"): (5.00e-6, 30.00e-6),
    ("openai", "gpt-5.4"): (2.50e-6, 15.00e-6), ("openai", "gpt-5.4-mini"): (0.75e-6, 4.50e-6),
    ("openai", "gpt-4.1"): (2.00e-6, 8.00e-6), ("openai", "gpt-4.1-mini"): (0.40e-6, 1.60e-6),
    ("anthropic", "claude-opus-5"): (5e-6, 25e-6),
    ("anthropic", "claude-opus-5-5"): (4e-6, 20e-6),
    ("anthropic", "claude-sonnet-5"): (2e-6, 10e-6),
    ("anthropic", "claude-haiku-4-5"): (1e-6, 5e-6),
}


@dataclass
class Completion:
    text: str
    tokens_in: int
    tokens_out: int
    latency: float
    cached: bool = False
    stop: str = ""


class _Cache:
    def __init__(self, provider, model):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        safe = f"{provider}_{model}".replace("/", "_").replace(":", "_")
        self.path = CACHE_DIR / f"{safe}.sqlite"
        with self._con() as c:
            c.execute("CREATE TABLE IF NOT EXISTS r (k TEXT PRIMARY KEY, v TEXT)")
            c.execute("CREATE TABLE IF NOT EXISTS spend (usd REAL)")

    def _con(self):
        return sqlite3.connect(self.path, timeout=60)

    def get(self, k):
        with self._con() as c:
            row = c.execute("SELECT v FROM r WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k, v, usd=0.0):
        with self._con() as c:
            c.execute("INSERT OR REPLACE INTO r VALUES (?, ?)", (k, json.dumps(v)))
            if usd:
                c.execute("INSERT INTO spend VALUES (?)", (usd,))

    def spent(self):
        with self._con() as c:
            return c.execute("SELECT COALESCE(SUM(usd), 0) FROM spend").fetchone()[0]


class SpendCapExceeded(RuntimeError):
    pass


class Provider:
    name = "base"

    def __init__(self, model, price=None):
        self.model = model
        self.price = price or PRICES.get((self.name, model), (0.0, 0.0))
        self.cache = _Cache(self.name, model)

    def complete(self, system: str, user: str, max_tokens=1024) -> Completion:
        k = hashlib.sha256(f"{self.model}\x00{system}\x00{user}".encode()).hexdigest()
        hit = self.cache.get(k)
        if hit:
            return Completion(**{**hit, "cached": True})
        cap = float(os.environ.get("ESAOG_SPEND_CAP_USD", "inf"))
        if self.price != (0.0, 0.0) and self.cache.spent() >= cap:
            raise SpendCapExceeded(f"{self.name}:{self.model} spend cap USD {cap} reached")
        t0 = time.perf_counter()
        text, tin, tout, stop = self._call(system, user, max_tokens)
        c = Completion(text, tin, tout, time.perf_counter() - t0, False, stop)
        usd = tin * self.price[0] + tout * self.price[1]
        self.cache.put(k, dict(text=c.text, tokens_in=tin, tokens_out=tout, latency=c.latency, stop=stop), usd)
        return c

    def _call(self, system, user, max_tokens):
        raise NotImplementedError


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model="claude-opus-5", effort="low", price=None):
        super().__init__(model, price)
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=6)
        self.effort = effort

    def _call(self, system, user, max_tokens):
        kw = dict(model=self.model, max_tokens=max(max_tokens, 4000), system=system,
                  cache_control={"type": "ephemeral"},  # automatic prefix caching (shared candidate lists)
                  messages=[{"role": "user", "content": user}])
        if self.model == "claude-haiku-4-5":
            kw["extra_body"] = {"temperature": 0}  # Haiku 4.5 honours temperature; SDK 1.x takes it via extra_body
        else:
            kw["output_config"] = {"effort": self.effort}
        r = self.client.messages.create(**kw)
        if r.stop_reason == "refusal":
            return "", r.usage.input_tokens, r.usage.output_tokens, "refusal"
        text = "".join(b.text for b in r.content if b.type == "text")
        return text, r.usage.input_tokens, r.usage.output_tokens, r.stop_reason or ""


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model, price=None):
        super().__init__(model, price)
        import openai
        self.client = openai.OpenAI(max_retries=6)
        self._temp_ok = True
        self.reasoning = os.environ.get("ESAOG_OPENAI_REASONING", "low") or None

    def _call(self, system, user, max_tokens):
        kw = dict(model=self.model, response_format={"type": "json_object"}, seed=0,
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        if self._temp_ok:
            kw["temperature"] = 0
        if self.reasoning:
            kw["reasoning_effort"] = self.reasoning
        for _ in range(3):  # drop parameters a given model does not support
            try:
                r = self.client.chat.completions.create(**kw)
                break
            except Exception as e:
                msg = str(e)
                if "temperature" in msg and "temperature" in kw:
                    self._temp_ok = False
                    kw.pop("temperature")
                elif "reasoning_effort" in msg and "reasoning_effort" in kw:
                    self.reasoning = None
                    kw.pop("reasoning_effort")
                else:
                    raise
        else:
            raise RuntimeError("OpenAI call failed after parameter fallbacks")
        u = r.usage
        return r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens, r.choices[0].finish_reason or ""


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, model="qwen2.5:14b", host="http://localhost:11434", num_ctx=8192, price=None):
        super().__init__(model, price)
        self.host, self.num_ctx = host, num_ctx

    def _call(self, system, user, max_tokens):
        body = dict(model=self.model, stream=False, format="json",
                    options=dict(temperature=0, seed=0, num_ctx=self.num_ctx, num_predict=max_tokens),
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as f:
            r = json.loads(f.read())
        return r["message"]["content"], r.get("prompt_eval_count", 0), r.get("eval_count", 0), r.get("done_reason", "")


def make_provider(spec: str) -> Provider:
    """spec = 'anthropic:claude-opus-5' | 'openai:<model>' | 'ollama:qwen2.5:14b'"""
    kind, _, model = spec.partition(":")
    if kind == "anthropic":
        return AnthropicProvider(model or "claude-opus-5", effort=os.environ.get("ESAOG_CLAUDE_EFFORT", "low"))
    if kind == "openai":
        if not model:
            raise SystemExit("openai provider needs an explicit model, e.g. openai:<model-id>")
        pin = os.environ.get("ESAOG_OPENAI_PRICE")  # "in_per_M,out_per_M"
        price = tuple(float(x) * 1e-6 for x in pin.split(",")) if pin else None
        return OpenAIProvider(model, price)
    if kind == "ollama":
        return OllamaProvider(model or "qwen2.5:14b", host=os.environ.get("ESAOG_OLLAMA_HOST", "http://localhost:11434"),
                              num_ctx=int(os.environ.get("ESAOG_OLLAMA_CTX", "8192")))
    raise SystemExit(f"unknown provider {spec}")
