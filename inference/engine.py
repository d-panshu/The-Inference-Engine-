"""
inference/engine.py
Phase 2 — Core inference engine.

Wraps Ollama via its Python SDK and exposes a clean async streaming interface.
vLLM continuous batching is layered on top in batching.py.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator

import httpx
import ollama

from monitoring.metrics import (
    INFERENCE_TTFT,
    INFERENCE_TOKENS_TOTAL,
    INFERENCE_REQUESTS_ACTIVE,
)


# ── Request / Response models ──────────────────────────────────────────────────

@dataclass
class InferenceRequest:
    messages: list[dict]               # [{role, content}, ...]
    model: str = "llama3:8b-instruct-q4_K_M"
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = True
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class TokenChunk:
    request_id: str
    token: str
    finish_reason: str | None = None   # None = more tokens, "stop"/"length" = done
    ttft_ms: float | None = None       # populated on the first chunk only


# ── Engine ─────────────────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Thin async wrapper around Ollama.

    Usage:
        engine = InferenceEngine()
        async for chunk in engine.generate(request):
            print(chunk.token, end="", flush=True)
    """

    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self._client = ollama.AsyncClient(host=ollama_host)
        self._ollama_host = ollama_host

    async def health_check(self) -> bool:
        """Returns True if Ollama is reachable and model is loaded."""
        try:
            async with httpx.AsyncClient(timeout=5) as http:
                r = await http.get(f"{self._ollama_host}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    async def generate(
        self, request: InferenceRequest
    ) -> AsyncGenerator[TokenChunk, None]:
        """
        Stream tokens from Ollama.

        Yields TokenChunk for every token. The first chunk includes ttft_ms.
        The final chunk has finish_reason set.

        Key concept — continuous batching:
            Ollama handles multiple concurrent calls. Each call to this method
            is independent; Ollama (and vLLM underneath) can interleave decode
            steps from different requests. This is "continuous batching" —
            request B doesn't wait for A to finish, it joins the next batch slot.
        """
        INFERENCE_REQUESTS_ACTIVE.inc()
        start_ns = time.perf_counter_ns()
        first_token = True
        token_count = 0

        try:
            async for part in await self._client.chat(
                model=request.model,
                messages=request.messages,
                stream=True,
                options={
                    "num_predict": request.max_tokens,
                    "temperature": request.temperature,
                },
            ):
                token = part["message"]["content"]
                done = part.get("done", False)
                token_count += 1

                ttft_ms = None
                if first_token:
                    ttft_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
                    INFERENCE_TTFT.observe(ttft_ms / 1000)  # histogram in seconds
                    first_token = False

                finish_reason = None
                if done:
                    finish_reason = "stop"
                    if part.get("eval_count", 0) >= request.max_tokens:
                        finish_reason = "length"

                yield TokenChunk(
                    request_id=request.request_id,
                    token=token,
                    finish_reason=finish_reason,
                    ttft_ms=ttft_ms,
                )

                if done:
                    break

        finally:
            INFERENCE_TOKENS_TOTAL.inc(token_count)
            INFERENCE_REQUESTS_ACTIVE.dec()

    async def generate_full(self, request: InferenceRequest) -> str:
        """Non-streaming: collect all tokens and return full text."""
        parts: list[str] = []
        async for chunk in self.generate(request):
            parts.append(chunk.token)
        return "".join(parts)


# ── KV Cache tracking (conceptual layer) ──────────────────────────────────────

class KVCacheTracker:
    """
    Tracks paged attention KV cache usage per request.

    In production vLLM manages this internally in 16 MB pages.
    This class mirrors the accounting so we can expose it to Prometheus.

    Key concept — paged attention:
        Instead of allocating a contiguous block of memory per request
        (which wastes RAM for variable-length sequences), vLLM allocates
        fixed-size "pages". Like virtual memory, pages are allocated on demand
        and freed when done. This lets many requests share the same physical
        RAM without fragmentation.
    """

    PAGE_SIZE_BYTES = 16 * 1024 * 1024  # 16 MB per page

    def __init__(self, total_ram_bytes: int = 4 * 1024 ** 3):
        self._total_pages = total_ram_bytes // self.PAGE_SIZE_BYTES
        self._used: dict[str, int] = {}  # request_id → pages used
        self._lock = asyncio.Lock()

    async def allocate(self, request_id: str, tokens: int) -> bool:
        """Returns True if allocation succeeded."""
        # ~2 floats per token per layer × 32 layers × 4096 hidden dim × 2 bytes
        pages_needed = max(1, (tokens * 32 * 128 * 2) // self.PAGE_SIZE_BYTES)
        async with self._lock:
            used_total = sum(self._used.values())
            if used_total + pages_needed > self._total_pages:
                return False
            self._used[request_id] = pages_needed
            return True

    async def free(self, request_id: str) -> None:
        async with self._lock:
            self._used.pop(request_id, None)

    @property
    def utilization(self) -> float:
        used = sum(self._used.values())
        return used / self._total_pages if self._total_pages else 0.0
