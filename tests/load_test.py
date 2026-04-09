"""
tests/load_test.py
Phase gate benchmark — run this after each phase to verify the gate criteria.

Usage:
    # Single request benchmark (Phase 2 gate)
    python tests/load_test.py --mode single

    # Concurrent load test (Phase 3 gate)
    python tests/load_test.py --mode concurrent --users 4

    # Full load test (Phase 4 gate)
    python tests/load_test.py --mode stress --users 10 --duration 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field

import httpx


BASE_URL = "http://localhost:8000"

PROMPT = {
    "model": "llama3",
    "messages": [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain what a transformer neural network is in 3 sentences."},
    ],
    "stream": False,
    "max_tokens": 150,
}


@dataclass
class RequestResult:
    request_id: str
    ttft_ms: float | None = None
    total_ms: float = 0.0
    tokens: int = 0
    error: str | None = None
    status_code: int = 200


async def single_request(client: httpx.AsyncClient) -> RequestResult:
    start = time.perf_counter()
    result = RequestResult(request_id=f"req-{int(start*1000)}")

    try:
        resp = await client.post(f"{BASE_URL}/v1/chat/completions", json=PROMPT, timeout=120)
        result.status_code = resp.status_code

        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            result.tokens = data["usage"]["completion_tokens"]
            result.total_ms = (time.perf_counter() - start) * 1000
            # Approximate TTFT — not exact without streaming, but useful
            result.ttft_ms = result.total_ms
        else:
            result.error = f"HTTP {resp.status_code}"

    except Exception as exc:
        result.error = str(exc)
        result.total_ms = (time.perf_counter() - start) * 1000

    return result


async def run_single():
    print("── Phase 2 gate: single request ──────────────────────")
    async with httpx.AsyncClient() as client:
        # Health check first
        health = await client.get(f"{BASE_URL}/health", timeout=5)
        print(f"Health: {health.json()}")
        print()

        result = await single_request(client)
        print(f"Status:       {result.status_code}")
        print(f"Total time:   {result.total_ms:.0f} ms")
        print(f"Tokens:       {result.tokens}")
        if result.tokens > 0:
            print(f"Throughput:   {result.tokens / (result.total_ms / 1000):.1f} tok/s")
        if result.error:
            print(f"Error:        {result.error}")
            sys.exit(1)

    gate_passed = result.total_ms < 10_000
    print()
    print(f"Gate (<10s): {'PASS ✓' if gate_passed else 'FAIL ✗'}")
    if not gate_passed:
        sys.exit(1)


async def run_concurrent(num_users: int):
    print(f"── Phase 3 gate: {num_users} concurrent requests ────────────")
    async with httpx.AsyncClient() as client:
        start = time.perf_counter()
        tasks = [single_request(client) for _ in range(num_users)]
        results: list[RequestResult] = await asyncio.gather(*tasks)
        wall_clock_ms = (time.perf_counter() - start) * 1000

    successes = [r for r in results if not r.error]
    errors = [r for r in results if r.error]
    latencies = [r.total_ms for r in successes]
    total_tokens = sum(r.tokens for r in successes)

    print(f"Requests:     {num_users}")
    print(f"Successes:    {len(successes)}")
    print(f"Errors:       {len(errors)}")
    if latencies:
        print(f"Latency P50:  {statistics.median(latencies):.0f} ms")
        print(f"Latency P99:  {sorted(latencies)[int(len(latencies)*0.99)]:.0f} ms")
    print(f"Wall clock:   {wall_clock_ms:.0f} ms")
    print(f"Total tokens: {total_tokens}")
    if wall_clock_ms > 0:
        print(f"Throughput:   {total_tokens / (wall_clock_ms / 1000):.1f} tok/s")

    gate_passed = len(errors) == 0
    print()
    print(f"Gate (0 errors): {'PASS ✓' if gate_passed else 'FAIL ✗'}")
    if not gate_passed:
        for r in errors:
            print(f"  Error: {r.error}")
        sys.exit(1)


async def run_stress(num_users: int, duration_s: int):
    print(f"── Phase 4 gate: stress test ({num_users} users, {duration_s}s) ──")
    all_results: list[RequestResult] = []
    end_time = time.monotonic() + duration_s

    async with httpx.AsyncClient() as client:
        while time.monotonic() < end_time:
            batch = [single_request(client) for _ in range(num_users)]
            results = await asyncio.gather(*batch)
            all_results.extend(results)

    successes = [r for r in all_results if not r.error]
    errors = [r for r in all_results if r.error]
    latencies = sorted(r.total_ms for r in successes)

    print(f"Total requests:  {len(all_results)}")
    print(f"Successes:       {len(successes)}")
    print(f"Errors:          {len(errors)} ({100*len(errors)/len(all_results):.1f}%)")
    if latencies:
        print(f"Latency P50:     {statistics.median(latencies):.0f} ms")
        print(f"Latency P99:     {latencies[int(len(latencies)*0.99)]:.0f} ms")
    print(f"Error rate gate (<5%): {'PASS ✓' if len(errors)/len(all_results) < 0.05 else 'FAIL ✗'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "concurrent", "stress"], default="single")
    parser.add_argument("--users", type=int, default=4)
    parser.add_argument("--duration", type=int, default=60)
    args = parser.parse_args()

    if args.mode == "single":
        asyncio.run(run_single())
    elif args.mode == "concurrent":
        asyncio.run(run_concurrent(args.users))
    elif args.mode == "stress":
        asyncio.run(run_stress(args.users, args.duration))
