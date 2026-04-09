"""
monitoring/metrics.py
Phase 5 — Prometheus instrumentation.

The five metrics that matter for an inference engine:
    1. TTFT        — Time To First Token (user-perceived latency)
    2. Throughput  — Tokens per second (system capacity)
    3. Batch size  — Active concurrent requests (utilisation)
    4. Queue depth — Waiting requests (backpressure signal)
    5. Memory      — RAM used by model + KV cache (capacity limit)

Exposed on /metrics in Prometheus text format.
Prometheus scrapes this every 15s (configured in prometheus.yml).
"""

from __future__ import annotations

import os
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response


# ── Inference engine metrics ───────────────────────────────────────────────────

INFERENCE_TTFT = Histogram(
    "llm_inference_ttft_seconds",
    "Time to first token in seconds",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 30.0],
)

INFERENCE_TOKENS_TOTAL = Counter(
    "llm_inference_tokens_total",
    "Total tokens generated across all requests",
)

INFERENCE_REQUESTS_ACTIVE = Gauge(
    "llm_inference_requests_active",
    "Number of requests currently in inference",
)

INFERENCE_REQUESTS_TOTAL = Counter(
    "llm_inference_requests_total",
    "Total inference requests completed",
    labelnames=["finish_reason"],  # stop | length | error
)


# ── Batching / scheduler metrics ───────────────────────────────────────────────

BATCH_SIZE_CURRENT = Gauge(
    "llm_batch_size_current",
    "Number of requests currently in a batch (continuous batching slot count)",
)

QUEUE_DEPTH = Gauge(
    "llm_queue_depth",
    "Number of requests waiting in the inference queue",
)

QUEUE_WAIT_SECONDS = Histogram(
    "llm_queue_wait_seconds",
    "Time a request waited in queue before inference started",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)


# ── Gateway metrics ────────────────────────────────────────────────────────────

GATEWAY_REQUESTS_TOTAL = Counter(
    "llm_gateway_requests_total",
    "Total HTTP requests received by the gateway",
    labelnames=["model", "stream"],
)

GATEWAY_LATENCY = Histogram(
    "llm_gateway_latency_seconds",
    "End-to-end request latency at the gateway",
    buckets=[1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0],
)

GATEWAY_ERRORS_TOTAL = Counter(
    "llm_gateway_errors_total",
    "Total gateway errors by type",
    labelnames=["error_type"],  # timeout | queue_full | model_error
)


# ── System metrics ─────────────────────────────────────────────────────────────

MEMORY_USAGE_BYTES = Gauge(
    "llm_memory_usage_bytes",
    "RAM currently consumed by model + KV cache",
)

KV_CACHE_UTILIZATION = Gauge(
    "llm_kv_cache_utilization_ratio",
    "Fraction of KV cache pages in use (0.0–1.0)",
)


# ── Metrics endpoint helper ────────────────────────────────────────────────────

def expose_metrics() -> Response:
    """Return Prometheus text format for /metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
