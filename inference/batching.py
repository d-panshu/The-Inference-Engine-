"""
inference/batching.py
Phase 2 — Continuous batching scheduler.

Key concept:
    Naive batching: collect N requests, run them together, return results.
    Problem: short requests wait for long ones to finish.

    Continuous batching (also called "in-flight batching"):
        After each decode step, check the queue for new requests.
        Evict finished sequences. Insert waiting sequences.
        Every decode step can have a different set of active requests.

    This file implements a simplified version of the vLLM scheduler.
    In production, vLLM's C++ scheduler does this at the token level.
    Here we do it at the request level for clarity.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator

from inference.engine import InferenceEngine, InferenceRequest, TokenChunk
from monitoring.metrics import BATCH_SIZE_CURRENT, QUEUE_DEPTH


@dataclass
class QueuedRequest:
    request: InferenceRequest
    result_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    enqueued_at: float = field(default_factory=time.monotonic)


class ContinuousBatchScheduler:
    """
    Manages a pool of concurrent inference slots.

    max_concurrent_requests:
        How many requests can be IN the engine simultaneously.
        On an 8-core CPU with Llama 3 8B, 4 is a practical ceiling.
        Each slot uses ~200MB of RAM for KV cache.

    max_queue_depth:
        Requests waiting for a slot. If full, return 429.

    The scheduler loop runs as a background asyncio task.
    It pulls from the waiting queue and fills available slots.
    When a slot finishes, it immediately offers the slot to the next waiter.
    This is the "continuous" part — there's no synchronisation barrier.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        max_concurrent: int = 4,
        max_queue_depth: int = 16,
    ):
        self._engine = engine
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue: asyncio.Queue[QueuedRequest] = asyncio.Queue(
            maxsize=max_queue_depth
        )
        self._running = False
        self._scheduler_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()

    async def submit(
        self, request: InferenceRequest
    ) -> AsyncGenerator[TokenChunk, None]:
        """
        Public entry point. Submit a request and async-iterate over token chunks.

        Raises QueueFullError if the system is at capacity.
        """
        queued = QueuedRequest(request=request)
        try:
            self._queue.put_nowait(queued)
            QUEUE_DEPTH.set(self._queue.qsize())
        except asyncio.QueueFull:
            raise QueueFullError("Inference queue full. Retry later.")

        # Stream tokens from the result queue
        while True:
            chunk: TokenChunk | Exception = await queued.result_queue.get()
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk
            if chunk.finish_reason is not None:
                break

    async def _scheduler_loop(self) -> None:
        """
        Continuous batching loop.

        Takes a request from the queue when a semaphore slot opens.
        Each request runs as an independent asyncio task so the loop
        can immediately pick up the next one (no serial waiting).
        """
        while self._running:
            try:
                queued: QueuedRequest = await asyncio.wait_for(
                    self._queue.get(), timeout=0.1
                )
                QUEUE_DEPTH.set(self._queue.qsize())

                # Acquire a slot — will wait if max_concurrent is saturated
                await self._semaphore.acquire()
                BATCH_SIZE_CURRENT.set(
                    self._max_concurrent - self._semaphore._value  # type: ignore[attr-defined]
                )

                # Spawn as a background task — don't await it here.
                # This is the key: the loop immediately goes back to check
                # the queue for the next request while this one runs.
                asyncio.create_task(self._run_request(queued))

            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                # Log and continue — scheduler must not crash
                print(f"Scheduler error: {exc}")

    async def _run_request(self, queued: QueuedRequest) -> None:
        """Run a single request to completion, forwarding chunks to the caller."""
        try:
            async for chunk in self._engine.generate(queued.request):
                await queued.result_queue.put(chunk)
        except Exception as exc:
            await queued.result_queue.put(exc)
        finally:
            self._semaphore.release()
            BATCH_SIZE_CURRENT.set(
                self._max_concurrent - self._semaphore._value  # type: ignore[attr-defined]
            )


class QueueFullError(Exception):
    pass
