"""
ray_cluster/bootstrap.py
Phase 3 — Cluster bootstrap script.

Run inside the ray-head container after `ray start --head` is healthy.
Initialises Ray, starts Ray Serve, and deploys InferenceDeployment.

The Docker entrypoint sequence:
    1. Container starts with `ray start --head --block` (keeps Ray running)
    2. A separate bootstrap job runs this script once Ray is healthy
    3. Ray Serve is now live on :8001, receiving traffic from the gateway

Usage (from docker-compose or manually):
    python ray_cluster/bootstrap.py

Environment variables consumed:
    RAY_ADDRESS         — GCS address (default: auto, discovers head)
    RAY_SERVE_PORT      — Ray Serve HTTP port (default: 8001)
    NUM_WORKER_REPLICAS — how many InferenceDeployment replicas (default: 1)
"""

from __future__ import annotations

import os
import sys
import time

import ray
from ray import serve

from ray_cluster.serve_deployment import InferenceDeployment


def wait_for_ray(max_retries: int = 30, delay: float = 2.0) -> None:
    """Poll until the Ray GCS is reachable."""
    address = os.getenv("RAY_ADDRESS", "auto")
    for attempt in range(max_retries):
        try:
            ray.init(address=address, ignore_reinit_error=True, log_to_driver=False)
            nodes = ray.nodes()
            alive = [n for n in nodes if n["Alive"]]
            if alive:
                print(f"[bootstrap] Ray cluster ready: {len(alive)} node(s) alive")
                return
        except Exception as exc:
            print(f"[bootstrap] Waiting for Ray... ({attempt+1}/{max_retries}): {exc}")
        time.sleep(delay)
    print("[bootstrap] ERROR: Ray cluster never became ready")
    sys.exit(1)


def deploy_serve(port: int = 8001, num_replicas: int = 1) -> None:
    """Start Ray Serve and deploy the InferenceDeployment."""
    serve.start(
        http_options={"host": "0.0.0.0", "port": port},
        detached=True,           # persists after this script exits
    )

    # Reconfigure replica count from env
    deployment = InferenceDeployment.options(num_replicas=num_replicas)

    handle = serve.run(
        deployment.bind(),
        route_prefix="/",
        name="inference",
    )

    print(f"[bootstrap] Ray Serve deployed on :{port}")
    print(f"[bootstrap] Replicas: {num_replicas}")
    print(f"[bootstrap] Health: http://localhost:{port}/health")


def main() -> None:
    serve_port = int(os.getenv("RAY_SERVE_PORT", "8001"))
    num_replicas = int(os.getenv("NUM_WORKER_REPLICAS", "1"))

    print("[bootstrap] Starting Ray cluster connection...")
    wait_for_ray()

    print("[bootstrap] Deploying Ray Serve...")
    deploy_serve(port=serve_port, num_replicas=num_replicas)

    print("[bootstrap] Done. Gateway can now route to Ray Serve.")


if __name__ == "__main__":
    main()
