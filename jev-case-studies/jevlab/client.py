"""Async clients for Jev (TypeSafe System One API) and for LLM comparators, both via Vercel AI Gateway.

Every request is cached on disk, keyed by a hash of (endpoint, model, payload), so re-running an
experiment never double-spends and every published number can be replayed from the cache.

Environment:
  AI_GATEWAY_API_KEY   Vercel AI Gateway key (default provider)
  JEV_PROVIDER         "gateway" (default), "typesafe" (direct, TYPESAFE_API_KEY) or
                       "openrouter" (OPENROUTER_API_KEY, model typesafe/jev-1.13)
  JEV_MOCK=1           Pipeline-test mode: deterministic fake answers, never real results.
                       Every artifact produced in this mode is stamped mock=true and the
                       report builder refuses to present it as a finding.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .paths import CACHE

GATEWAY_BASE = "https://ai-gateway.vercel.sh"
JEV_PRICE_PER_TOKEN = 0.042 / 1_000_000  # input only; output tokens are free

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def is_mock() -> bool:
    return os.environ.get("JEV_MOCK") == "1"


def _key(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class DiskCache:
    """Append-only JSONL cache. One file per namespace; safe for a single writer process."""

    def __init__(self, namespace: str):
        suffix = ".mock" if is_mock() else ""
        self.path = CACHE / f"{namespace}{suffix}.jsonl"
        self.mem: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn final line from an interrupted run
                    self.mem[rec["key"]] = rec
        self._fh = self.path.open("a")

    def get(self, key: str) -> dict | None:
        return self.mem.get(key)

    def put(self, key: str, rec: dict) -> None:
        rec = {"key": key, **rec}
        self.mem[key] = rec
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


@dataclass
class CallStats:
    calls: int = 0
    cached: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        lat = sorted(self.latencies_ms)
        pct = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] if lat else None  # noqa: E731
        return {
            "calls": self.calls,
            "cached": self.cached,
            "errors": self.errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms_p50": pct(0.5),
            "latency_ms_p90": pct(0.9),
            "latency_ms_p99": pct(0.99),
        }


class _Base:
    def __init__(self, namespace: str, concurrency: int = 16, timeout_s: float = 60.0):
        self.cache = DiskCache(namespace)
        self.sem = asyncio.Semaphore(concurrency)
        # Start-spacing rate limiter (TypeSafe documents 1,200 requests/minute; stay under it).
        self._min_interval = 60.0 / float(os.environ.get("JEV_MAX_RPM", "1000"))
        self._next_slot = 0.0
        self._rl_lock = asyncio.Lock()
        self.stats = CallStats()
        self.timeout_s = timeout_s
        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._http = httpx.AsyncClient(timeout=self.timeout_s, http2=False)
        return self

    async def __aexit__(self, *exc):
        if self._http:
            await self._http.aclose()
        self.cache.close()

    async def _post(self, url: str, headers: dict, payload: dict) -> tuple[dict, float]:
        """POST with exponential backoff. Returns (json, latency_ms of the successful attempt)."""
        assert self._http is not None
        delay = 1.0
        for attempt in range(8):
            async with self._rl_lock:
                now = time.monotonic()
                wait = self._next_slot - now
                self._next_slot = max(now, self._next_slot) + self._min_interval
            if wait > 0:
                await asyncio.sleep(wait)
            t0 = time.perf_counter()
            try:
                r = await self._http.post(url, headers=headers, json=payload)
            except (httpx.TransportError, httpx.TimeoutException):
                if attempt == 7:
                    raise
                await asyncio.sleep(delay + random.random())
                delay = min(delay * 2, 30)
                continue
            latency_ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                return r.json(), latency_ms
            if r.status_code in RETRY_STATUS and attempt < 7:
                ra = r.headers.get("retry-after")
                wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay
                await asyncio.sleep(wait + random.random())
                delay = min(delay * 2, 30)
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        raise RuntimeError("exhausted retries")


class JevClient(_Base):
    """Calls POST /v1/systemone with TypeSafe's request/response shapes."""

    def __init__(self, namespace: str, concurrency: int = 16, model: str | None = None):
        super().__init__(namespace, concurrency)
        provider = os.environ.get("JEV_PROVIDER", "gateway")
        if provider == "openrouter":
            self.url = "https://openrouter.ai/api/v1/systemone"
            self.model = model or os.environ.get("JEV_MODEL", "typesafe/jev-1.13")
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        elif provider == "typesafe":
            self.url = "https://api.typesafe.ai/v1/systemone"
            self.model = model or os.environ.get("JEV_MODEL", "jev-latest")
            self.api_key = os.environ.get("TYPESAFE_API_KEY", "")
        else:
            self.url = f"{GATEWAY_BASE}/typesafe/v1/systemone"
            self.model = model or os.environ.get("JEV_MODEL", "typesafe-ai/jev")
            self.api_key = os.environ.get("AI_GATEWAY_API_KEY", "")
        if not self.api_key and not is_mock():
            raise SystemExit(
                "No API key. Set AI_GATEWAY_API_KEY (Vercel AI Gateway) or run with JEV_MOCK=1 "
                "to test the pipeline without calling Jev."
            )

    async def ask(self, state: Any, questions: dict, *, sequential_latency: bool = False) -> dict:
        """Returns {"answers", "usage", "latency_ms", "model", "cost_usd", "cached"}.

        sequential_latency=True bypasses the concurrency pool (one request in flight) so the
        recorded latency is a clean round trip, used for the real-time latency measurements.
        """
        payload = {"model": self.model, "state": state, "questions": questions}
        key = _key({"url": self.url, "payload": payload, "seq": sequential_latency})
        hit = self.cache.get(key)
        if hit is not None:
            self.stats.cached += 1
            self._account(hit)
            return {**hit, "cached": True}

        if is_mock():
            rec = _mock_answer(payload)
        else:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            if sequential_latency:
                data, lat = await self._post(self.url, headers, payload)
            else:
                async with self.sem:
                    data, lat = await self._post(self.url, headers, payload)
            usage = data.get("usage") or {}
            gw_cost = (((data.get("provider_metadata") or {}).get("gateway") or {}).get("cost"))
            if gw_cost is None:
                gw_cost = usage.get("cost")  # OpenRouter reports cost inside usage
            in_tok = int(usage.get("input_tokens") or 0)
            rec = {
                "answers": data.get("answers", {}),
                "usage": usage,
                "latency_ms": lat,
                "model": data.get("model", self.model),
                "cost_usd": float(gw_cost) if gw_cost is not None else in_tok * JEV_PRICE_PER_TOKEN,
                "mock": False,
                "ts": time.time(),
            }
        self.cache.put(key, rec)
        self.stats.calls += 1
        self._account(rec)
        return {**rec, "cached": False}

    def _account(self, rec: dict) -> None:
        u = rec.get("usage") or {}
        self.stats.input_tokens += int(u.get("input_tokens") or 0)
        self.stats.output_tokens += int(u.get("output_tokens") or 0)
        self.stats.cost_usd += float(rec.get("cost_usd") or 0)
        if rec.get("latency_ms") is not None:
            self.stats.latencies_ms.append(float(rec["latency_ms"]))


def _mock_answer(payload: dict) -> dict:
    """Deterministic fake answers with the documented response shapes (pipeline testing only)."""
    rng = random.Random(_key(payload))
    answers = {}
    for qid, q in payload["questions"].items():
        t = q.get("type")
        if t == "noul":
            answers[qid] = {"type": "noul", "noul": round(rng.random(), 4)}
        elif t == "choice":
            opts = list(q["criteria"].keys())
            w = [rng.random() ** 3 for _ in opts]
            s = sum(w)
            probs = {o: x / s for o, x in zip(opts, w)}
            best = max(probs, key=probs.get)
            answers[qid] = {"type": "choice", "choice": best, "probabilities": probs,
                            "confidence": round(max(probs.values()), 4)}
        elif t == "score":
            n = len(q["criteria"])
            w = [rng.random() ** 3 for _ in range(n)]
            s = sum(w)
            probs = {str(i): x / s for i, x in enumerate(w)}
            score = sum(i * p for i, p in enumerate(probs.values()))
            answers[qid] = {"type": "score", "score": score, "probabilities": probs,
                            "legend": {str(i): str(c) for i, c in enumerate(q["criteria"])},
                            "confidence": round(max(probs.values()), 4)}
    in_tok = len(json.dumps(payload)) // 4
    return {"answers": answers, "usage": {"input_tokens": in_tok, "output_tokens": 0},
            "latency_ms": max(40.0, rng.gauss(180, 60)), "model": "MOCK", "mock": True,
            "cost_usd": in_tok * JEV_PRICE_PER_TOKEN, "ts": time.time()}


class LLMClient(_Base):
    """OpenAI-compatible chat completions through Vercel AI Gateway, used as a System-Two comparator.

    The model is asked for a single probability as JSON. Latency is end-to-end, and cost uses the
    gateway's public per-token list price for the model (fetched once and saved with the results).
    """

    def __init__(self, namespace: str, model: str, concurrency: int = 8, pricing: dict | None = None):
        super().__init__(namespace, concurrency, timeout_s=120.0)
        self.model = model
        self.url = f"{GATEWAY_BASE}/v1/chat/completions"
        self.api_key = os.environ.get("AI_GATEWAY_API_KEY", "")
        self.pricing = pricing or {}
        if not self.api_key and not is_mock():
            raise SystemExit("No AI_GATEWAY_API_KEY set for the LLM comparator.")

    async def probability(self, system: str, user: str, *, sequential_latency: bool = False) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": 40,
        }
        key = _key({"url": self.url, "payload": payload, "seq": sequential_latency})
        hit = self.cache.get(key)
        if hit is not None:
            self.stats.cached += 1
            self._account(hit)
            return {**hit, "cached": True}
        if is_mock():
            rng = random.Random(key)
            rec = {"p": rng.random(), "raw": "{}", "usage": {"input_tokens": len(user) // 4, "output_tokens": 8},
                   "latency_ms": max(200.0, rng.gauss(900, 300)), "mock": True, "cost_usd": 0.0}
        else:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            if sequential_latency:
                data, lat = await self._post(self.url, headers, payload)
            else:
                async with self.sem:
                    data, lat = await self._post(self.url, headers, payload)
            text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            u = data.get("usage") or {}
            in_tok = int(u.get("prompt_tokens") or 0)
            out_tok = int(u.get("completion_tokens") or 0)
            price_in = float(self.pricing.get("input") or 0)
            price_out = float(self.pricing.get("output") or 0)
            rec = {"p": parse_probability(text), "raw": text[:200],
                   "usage": {"input_tokens": in_tok, "output_tokens": out_tok}, "latency_ms": lat,
                   "mock": False, "cost_usd": in_tok * price_in + out_tok * price_out, "ts": time.time()}
        self.cache.put(key, rec)
        self.stats.calls += 1
        self._account(rec)
        return {**rec, "cached": False}

    def _account(self, rec: dict) -> None:
        u = rec.get("usage") or {}
        self.stats.input_tokens += int(u.get("input_tokens") or 0)
        self.stats.output_tokens += int(u.get("output_tokens") or 0)
        self.stats.cost_usd += float(rec.get("cost_usd") or 0)
        if rec.get("latency_ms") is not None:
            self.stats.latencies_ms.append(float(rec["latency_ms"]))
        if rec.get("p") is None:
            self.stats.errors += 1


def parse_probability(text: str) -> float | None:
    """Extract a probability from '{"probability": 0.7}' or a bare number. None if unparseable."""
    import re

    try:
        obj = json.loads(text.strip().strip("`").removeprefix("json").strip())
        if isinstance(obj, dict):
            for k in ("probability", "p", "prob"):
                if k in obj:
                    v = float(obj[k])
                    return min(1.0, max(0.0, v / 100 if v > 1 else v))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    m = re.search(r"(\d*\.?\d+)\s*(%)?", text)
    if not m:
        return None
    v = float(m.group(1))
    if m.group(2) or v > 1:
        v = v / 100
    return v if 0 <= v <= 1 and not math.isnan(v) else None


def fetch_gateway_pricing(model: str) -> dict:
    """Public list price for a gateway model (USD per token)."""
    r = httpx.get(f"{GATEWAY_BASE}/v1/models", timeout=30)
    r.raise_for_status()
    for m in r.json().get("data", []):
        if m.get("id") == model:
            return m.get("pricing") or {}
    return {}


# Answer accessors -------------------------------------------------------------------------------

def noul(ans: dict, qid: str) -> float:
    return float(ans["answers"][qid]["noul"])


def score(ans: dict, qid: str) -> float:
    return float(ans["answers"][qid]["score"])


def choice_probs(ans: dict, qid: str) -> dict[str, float]:
    return {k: float(v) for k, v in ans["answers"][qid]["probabilities"].items()}


def run_all(coros, desc: str = "", every: int = 200):
    """Gather coroutines with a lightweight progress line."""

    async def _runner():
        done = 0
        results = [None] * len(coros)

        async def wrap(i, c):
            nonlocal done
            results[i] = await c
            done += 1
            if every and done % every == 0:
                print(f"  [{desc}] {done}/{len(coros)}", flush=True)

        await asyncio.gather(*(wrap(i, c) for i, c in enumerate(coros)))
        return results

    return _runner()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))
