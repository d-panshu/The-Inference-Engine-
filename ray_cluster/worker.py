"""
ray_cluster/worker.py
Phase 3 — Ray Actor: the actual compute unit inside the cluster.

Each InferenceWorker is a Ray Actor — an isolated stateful process
that Ray schedules on available CPUs. Workers are concurrent and
communicate via Ray's object store (Apache Arrow serialisation, no shared mem).

Tensor parallelism:
    Real GPU clusters (Megatron-LM, DeepSpeed, vLLM):
        - Attention Q/K/V weight matrices split column-wise across GPUs
        - Each GPU computes attention for its assigned heads, then all-reduce
        - Feed-forward split row-wise
    CPU simulation:
        - Worker 1 conceptually owns layers 0–15 (first half of transformer)
        - Worker 2 conceptually owns layers 16–31 (second half)
        - Ollama runs the full model (can't split without vLLM on GPU)
        - The actor architecture, scheduling, and communication pattern
          are production-identical — only the compute is unified
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field

import ray

from inference.engine import InferenceEngine, InferenceRequest
from inference.batching import ContinuousBatchScheduler, QueueFullError
from inference.kv_cache import PagedKVCacheManager
from monitoring.metrics import (
    MEMORY_USAGE_BYTES,
    KV_CACHE_UTILIZATION,
    INFERENCE_REQUESTS_TOTAL,
)


MODEL_ALIASES = {
    "llama3":   "llama3:8b-instruct-q4_K_M",
    "mistral":  "mistral:7b-instruct-v0.2-q4_K_M",
    "llama2":   "llama2:7b-chat-q4_K_M",
}


@dataclass
class WorkerStats:
    worker_id: int
    shard_range: tuple[int, int]
    requests_completed: int
    requests_failed: int
    avg_ttft_ms: float
    queue_depth: int
    kv_utilization_pct: float
    ollama_healthy: bool


@ray.remote(num_cpus=2, max_restarts=3)
class InferenceWorker:
    """
    Ray Actor that owns:
        - One InferenceEngine  (wraps Ollama)
        - One ContinuousBatchScheduler  (manages concurrent slots)
        - One PagedKVCacheManager  (tracks KV page allocation)

    max_restarts=3:
        If this actor crashes (OOM, etc.), Ray auto-restarts up to 3 times.
        The Ray head node detects death via heartbeat within ~5s and respawns.

    num_cpus=2:
        Ray reserves 2 CPUs for this actor. Layout with 8 total cores:
            head node process  → 1 CPU
            worker-1 actor     → 2 CPUs  (this class)
            worker-2 actor     → 2 CPUs
            OS + gateway       → 3 CPUs
    """

    def __init__(self, worker_id: int, shard_range: tuple[int, int]):
        self.worker_id = worker_id
        self.shard_range = shard_range

        # Stats tracking
        self._requests_completed = 0
        self._requests_failed = 0
        self._ttft_samples: list[float] = []

        # Read config from env
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        max_concurrent = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))
        max_queue = int(os.getenv("MAX_QUEUE_DEPTH", "8"))
        kv_ram = int(os.getenv("KV_CACHE_RAM_BYTES", str(4 * 1024 ** 3)))

        self._default_model = os.getenv(
            "OLLAMA_MODEL", "llama3:8b-instruct-q4_K_M"
        )

        # Core components
        self._engine = InferenceEngine(ollama_host=ollama_host)
        self._scheduler = ContinuousBatchScheduler(
            engine=self._engine,
            max_concurrent=max_concurrent,
            max_queue_depth=max_queue,
        )
        self._kv_cache = PagedKVCacheManager(total_ram_bytes=kv_ram)

        # Start scheduler background loop
        asyncio.get_event_loop().run_until_complete(self._scheduler.start())

        print(
            f"[Worker {worker_id}] Ready — "
            f"shard layers {shard_range[0]}–{shard_range[1]}, "
            f"slots={max_concurrent}, queue_max={max_queue}, "
            f"kv_pages={self._kv_cache._total}"
        )

    # ── Primary inference method ───────────────────────────────────────────────

    async def generate(self, request_dict: dict) -> dict:
        """
        Run one inference request. Called via ray.remote:
            result = await worker.generate.remote(payload)

        Ray serialises the dict via Apache Arrow through its object store.
        This is zero-copy for numpy arrays; dicts/strings use pickle fallback.
        """
        request_dict["model"] = self._resolve_model(request_dict.get("model", "llama3"))
        request = InferenceRequest(**{
            k: v for k, v in request_dict.items()
            if k in InferenceRequest.__dataclass_fields__
        })

        # Register request in KV cache page allocator
        kv_ok = await self._kv_cache.register_request(request.request_id)
        if not kv_ok:
            self._requests_failed += 1
            return {
                "error": "kv_cache_full",
                "message": "No KV cache pages available. Server at capacity.",
                "status": 503,
                "worker_id": self.worker_id,
            }

        tokens: list[str] = []
        ttft_ms: float | None = None
        finish_reason = "stop"

        try:
            async for chunk in self._scheduler.submit(request):
                tokens.append(chunk.token)
                if chunk.ttft_ms is not None:
                    ttft_ms = chunk.ttft_ms
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                # Track KV page growth per token
                await self._kv_cache.on_token_generated(request.request_id)

            self._requests_completed += 1
            INFERENCE_REQUESTS_TOTAL.labels(finish_reason=finish_reason).inc()
            if ttft_ms is not None:
                self._ttft_samples.append(ttft_ms)
                if len(self._ttft_samples) > 200:
                    self._ttft_samples = self._ttft_samples[-200:]

            self._push_metrics()

            return {
                "request_id": request.request_id,
                "worker_id": self.worker_id,
                "shard": f"layers {self.shard_range[0]}–{self.shard_range[1]}",
                "tokens": tokens,
                "text": "".join(tokens),
                "finish_reason": finish_reason,
                "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
                "token_count": len(tokens),
            }

        except QueueFullError:
            self._requests_failed += 1
            return {
                "error": "queue_full",
                "message": "Inference queue full. Retry after 1s.",
                "status": 429,
                "worker_id": self.worker_id,
            }

        except Exception as exc:
            self._requests_failed += 1
            INFERENCE_REQUESTS_TOTAL.labels(finish_reason="error").inc()
            return {
                "error": str(exc),
                "status": 500,
                "worker_id": self.worker_id,
            }

        finally:
            await self._kv_cache.free_request(request.request_id)

    # ── Health & stats ─────────────────────────────────────────────────────────

    async def health(self) -> dict:
        """Lightweight probe — used by Ray Serve's health check interval."""
        return {
            "worker_id": self.worker_id,
            "healthy": await self._engine.health_check(),
            "shard": f"layers {self.shard_range[0]}–{self.shard_range[1]}",
            "kv_utilization_pct": round(self._kv_cache.utilization * 100, 1),
            "queue_depth": self._scheduler._queue.qsize(),
        }

    async def stats(self) -> WorkerStats:
        """Full stats — called by monitoring loop every 30s."""
        avg_ttft = (
            sum(self._ttft_samples) / len(self._ttft_samples)
            if self._ttft_samples else 0.0
        )
        return WorkerStats(
            worker_id=self.worker_id,
            shard_range=self.shard_range,
            requests_completed=self._requests_completed,
            requests_failed=self._requests_failed,
            avg_ttft_ms=round(avg_ttft, 1),
            queue_depth=self._scheduler._queue.qsize(),
            kv_utilization_pct=round(self._kv_cache.utilization * 100, 1),
            ollama_healthy=await self._engine.health_check(),
        )

    async def kv_stats(self) -> dict:
        """Return KV cache page table stats."""
        return self._kv_cache.stats()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _resolve_model(self, alias: str) -> str:
        """Map short model name → full Ollama model tag."""
        return MODEL_ALIASES.get(alias, alias) or self._default_model

    def _push_metrics(self) -> None:
        """Push current resource usage to Prometheus gauges."""
        try:
            import psutil, os as _os
            mem = psutil.Process(_os.getpid()).memory_info().rss
            MEMORY_USAGE_BYTES.set(mem)
        except Exception:
            pass
        KV_CACHE_UTILIZATION.set(self._kv_cache.utilization)
