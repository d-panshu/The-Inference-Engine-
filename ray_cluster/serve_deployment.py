"""
ray_cluster/serve_deployment.py
Phase 3 — Ray Serve: distributed orchestration layer.

Key concepts:
    @serve.deployment:
        Wraps a Python class into a Ray Serve deployment.
        Ray manages replicas, health checks, and traffic routing.

    num_replicas:
        How many copies of InferenceDeployment run simultaneously.
        Each replica is an independent process (or thread in local mode).
        Ray load-balances incoming requests across replicas.

    ray.remote (InferenceWorker):
        The actual inference work runs inside a Ray Actor.
        Actors are stateful workers that Ray schedules on available CPUs.
        Each actor holds its own InferenceEngine + ContinuousBatchScheduler.

    Tensor parallelism simulation:
        On a real GPU cluster, tensor parallelism splits the weight matrices
        across devices (e.g. attention heads on GPU 0, feed-forward on GPU 1).
        On our CPU-only setup, we simulate this by assigning layer ranges to
        workers. Ollama runs the full model, but workers report which "shard"
        they're responsible for — this demonstrates the concept for the resume.
"""

from __future__ import annotations

import os
import asyncio
from typing import AsyncGenerator

import ray
from ray import serve

from inference.engine import InferenceEngine, InferenceRequest
from inference.batching import ContinuousBatchScheduler, QueueFullError


# ── Ray Actor: one per CPU worker ─────────────────────────────────────────────

@ray.remote(num_cpus=2)
class InferenceWorker:
    """
    Ray Actor that owns an InferenceEngine and a ContinuousBatchScheduler.

    Runs in its own process (Ray isolates actors). Communicates via
    Ray's object store — no shared memory, no race conditions.

    shard_id:
        In a real tensor-parallel setup, this worker handles a subset
        of the model's weight matrices. Here it's metadata only, but the
        architecture is production-identical.
    """

    def __init__(self, worker_id: int, shard_range: tuple[int, int]):
        self.worker_id = worker_id
        self.shard_range = shard_range  # (start_layer, end_layer)
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._engine = InferenceEngine(ollama_host=ollama_host)
        self._scheduler = ContinuousBatchScheduler(
            engine=self._engine,
            max_concurrent=2,   # 2 slots per worker × 2 workers = 4 total
            max_queue_depth=8,
        )
        # Start the scheduler loop in the actor's event loop
        asyncio.get_event_loop().run_until_complete(self._scheduler.start())

    async def generate_stream(self, request_dict: dict) -> list[str]:
        """
        Ray remote methods can't return async generators directly.
        We collect all tokens and return a list.
        For real streaming, use Ray streaming (ray.experimental.channel).
        """
        request = InferenceRequest(**request_dict)
        tokens: list[str] = []
        async for chunk in self._scheduler.submit(request):
            tokens.append(chunk.token)
        return tokens

    async def health(self) -> dict:
        ok = await self._engine.health_check()
        return {
            "worker_id": self.worker_id,
            "shard_range": self.shard_range,
            "ollama_healthy": ok,
        }


# ── Ray Serve Deployment: HTTP entry point ─────────────────────────────────────

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1},
    max_concurrent_queries=16,
)
class InferenceDeployment:
    """
    Ray Serve deployment — the HTTP router inside the Ray cluster.

    Manages a pool of InferenceWorker actors and load-balances across them.
    Gateway (FastAPI) calls this via HTTP on port 8001.
    """

    def __init__(self):
        # Spawn two workers covering different "layer shards"
        self._workers = [
            InferenceWorker.remote(worker_id=1, shard_range=(0, 15)),
            InferenceWorker.remote(worker_id=2, shard_range=(16, 31)),
        ]
        self._rr_counter = 0  # round-robin index

    def _next_worker(self) -> "ray.actor.ActorHandle":
        """Simple round-robin load balancing."""
        worker = self._workers[self._rr_counter % len(self._workers)]
        self._rr_counter += 1
        return worker

    async def __call__(self, request) -> dict:
        """
        Handle an HTTP request from the gateway.
        Returns {"tokens": [...], "request_id": "..."} as JSON.
        """
        body = await request.json()
        worker = self._next_worker()

        try:
            # ray.remote call — runs on an actor process
            tokens: list[str] = await worker.generate_stream.remote(body)
            return {
                "request_id": body.get("request_id", ""),
                "tokens": tokens,
                "text": "".join(tokens),
            }
        except QueueFullError:
            return {"error": "queue_full", "status": 429}
        except Exception as exc:
            return {"error": str(exc), "status": 500}

    async def health(self) -> list[dict]:
        results = await asyncio.gather(
            *[w.health.remote() for w in self._workers]
        )
        return list(results)


# ── Cluster bootstrap ──────────────────────────────────────────────────────────

def start_serve():
    """
    Called once at startup to initialise the Ray cluster and deploy.
    Run this from the ray-head container after `ray start --head`.
    """
    ray_address = os.getenv("RAY_ADDRESS", "auto")
    ray.init(address=ray_address, ignore_reinit_error=True)

    serve.start(
        http_options={"host": "0.0.0.0", "port": 8001},
        detached=True,
    )

    deployment = InferenceDeployment.bind()
    serve.run(deployment, route_prefix="/")
    print("Ray Serve deployment running on :8001")


if __name__ == "__main__":
    start_serve()
