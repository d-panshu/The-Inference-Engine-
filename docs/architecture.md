# Distributed LLM Inference Engine — Architecture Document

**Version:** 1.0  
**Status:** Approved (Phase 0 Gate)  
**Hardware:** Intel Core i5-1135G7 · 16 GB RAM · Ubuntu 24.04  
**Author:** Engineering  

---

## 1. Problem Statement

Run a production-grade LLM inference engine on a single commodity laptop,
simulating a distributed multi-node cluster using Ray across 8 CPU cores.
The system must serve concurrent requests via an OpenAI-compatible REST API,
with live observability via Prometheus and Grafana.

---

## 2. Goals & Non-Goals

### Goals
- Serve Llama 3 (8B) or Mistral (7B) locally via Ollama
- OpenAI-compatible `/v1/chat/completions` endpoint (streaming + non-streaming)
- Distribute inference work across 8 CPU cores using Ray
- Continuous batching so concurrent requests don't queue serially
- Live Grafana dashboard: TTFT, tokens/sec, queue depth, memory, batch size
- All services containerised with Docker Compose

### Non-Goals
- GPU acceleration (no NVIDIA GPU on target hardware)
- Multi-machine deployment (single laptop, Ray simulates distribution)
- Fine-tuning or training
- Authentication / multi-tenancy (out of scope v1)

---

## 3. Capacity Planning

| Resource        | Budget              | Reasoning                                      |
|-----------------|---------------------|------------------------------------------------|
| Model weights   | 5–8 GB RAM          | Llama 3 8B in Q4 quant ≈ 5 GB                 |
| Ray overhead    | 1–2 GB RAM          | Head node + 3 workers                          |
| OS + services   | 2 GB RAM            | Prometheus, Grafana, FastAPI                   |
| **Total**       | **≤ 14 GB / 16 GB** | 2 GB headroom                                  |
| CPU cores       | 8 threads           | 1 head node + up to 7 Ray worker processes     |
| Disk            | ~10 GB              | Model weights volume mount                     |

**Latency budget (CPU-only targets):**

| Metric                | Target      |
|-----------------------|-------------|
| Time To First Token   | < 5 s       |
| Throughput            | > 5 tok/s   |
| P99 latency (1 user)  | < 10 s      |
| Concurrent requests   | 4 sustained |

---

## 4. High-Level Architecture

```
┌─────────────┐     HTTP/SSE      ┌──────────────────┐     gRPC
│   Client    │ ────────────────► │  FastAPI Gateway  │ ──────────────►
│ (curl / SDK)│                   │  Rate limit       │
└─────────────┘                   │  Auth middleware  │
                                  │  Request queue    │
                                  └──────────────────┘
                                           │
                                    Ray Serve HTTP
                                           │
                          ┌────────────────▼─────────────────┐
                          │         Ray Cluster              │
                          │  ┌─────────┐                     │
                          │  │  Head   │  Scheduler          │
                          │  │  Node   │  Object store       │
                          │  └────┬────┘                     │
                          │       │  ray.remote dispatch     │
                          │  ┌────▼──┐  ┌────────┐  ┌─────┐ │
                          │  │ Work  │  │ Work   │  │ KV  │ │
                          │  │  er 1 │  │  er 2  │  │Cache│ │
                          │  │Shard  │  │ Shard  │  │     │ │
                          │  │ 0–15  │  │ 16–31  │  │     │ │
                          │  └────┬──┘  └───┬────┘  └─────┘ │
                          └───────┼──────────┼───────────────┘
                                  └────┬─────┘
                                       │
                          ┌────────────▼──────────────┐
                          │   Ollama + vLLM Engine     │
                          │   Llama 3 8B               │
                          │   Continuous batching      │
                          │   Paged attention (KV)     │
                          └────────────┬──────────────┘
                                       │  streamed tokens
                                       ▼
                                   Client SSE

Observability (sidecar to all services):
  Prometheus ◄── scrape /metrics ── all containers
  Grafana ◄──── query ── Prometheus
```

---

## 5. Request Lifecycle (Sequence Diagram)

```
Client          Gateway         Ray Serve       Worker          vLLM
  │                │                │              │               │
  │ POST /v1/chat  │                │              │               │
  │───────────────►│                │              │               │
  │                │ validate req   │              │               │
  │                │ rate limit     │              │               │
  │                │────────────────►              │               │
  │                │                │ dispatch     │               │
  │                │                │─────────────►│               │
  │                │                │              │ batch insert  │
  │                │                │              │──────────────►│
  │                │                │              │               │ infer
  │                │                │              │◄── tokens ────│
  │◄── SSE chunk ──│◄── stream ─────│◄── stream ───│               │
  │◄── SSE chunk ──│                │              │               │
  │◄── [DONE] ─────│                │              │               │
```

---

## 6. Component Contracts

### 6.1 FastAPI Gateway

- Listens on `0.0.0.0:8000`
- Routes: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`, `GET /metrics`
- Validates OpenAI-schema request via Pydantic v2
- Enforces token-bucket rate limit: 10 req/min per IP (configurable)
- Forwards to Ray Serve via HTTP on `ray-serve:8001`

### 6.2 Ray Serve Deployment

- Listens on `0.0.0.0:8001` (internal only)
- Manages `num_replicas` based on CPU count
- Routes to `InferenceWorker` actors
- Handles backpressure: max queue depth = 16 requests

### 6.3 Inference Worker (Ray Actor)

- Loads model shard assigned at startup
- Calls vLLM `AsyncLLMEngine` for continuous batching
- Returns `AsyncGenerator[str, None]` of token strings
- Reports metrics to Prometheus push gateway

### 6.4 Ollama / vLLM Engine

- Model: `llama3:8b-instruct-q4_K_M` (quantised, fits in 5 GB)
- Paged attention: KV cache managed in 16 MB pages
- Continuous batching: new requests inserted between decode steps
- Context length: 4096 tokens max

---

## 7. API Contract (OpenAI-compatible)

```yaml
POST /v1/chat/completions
Content-Type: application/json

Request:
  model: string          # "llama3" or "mistral"
  messages: [{role, content}]
  stream: bool           # true = SSE, false = JSON response
  max_tokens: int        # default 512
  temperature: float     # default 0.7

Response (stream=false):
  id: string
  object: "chat.completion"
  choices: [{message: {role, content}, finish_reason}]
  usage: {prompt_tokens, completion_tokens, total_tokens}

Response (stream=true):
  data: {"choices": [{"delta": {"content": "..."}}]}\n\n
  data: [DONE]\n\n
```

---

## 8. Data Flow & Storage

- **Model weights**: Docker volume `llm-weights` mounted at `/root/.ollama` in the Ollama container. Persists across restarts.
- **KV cache**: In-memory only. Evicted on request completion.
- **Metrics**: Prometheus TSDB in `monitoring-data` volume. 15-day retention.
- **Logs**: stdout (JSON structured). Collected by Docker logging driver.

---

## 9. Failure Modes & Mitigations

| Failure                  | Detection              | Mitigation                        |
|--------------------------|------------------------|-----------------------------------|
| OOM (model too large)    | Container OOM kill     | Use Q4 quant; memory limit alerts |
| Worker crash             | Ray actor restart      | Ray auto-restarts actors (max 3x) |
| Request timeout          | Gateway 30s deadline   | Return 504, log trace ID          |
| Queue overflow           | Depth > 16             | Return 429 with Retry-After       |
| Ollama model not loaded  | Health check fails     | Docker depends_on + healthcheck   |

---

## 10. Phase Gate Checklist

- [ ] This document reviewed and approved
- [ ] Capacity plan fits within 16 GB RAM
- [ ] API contract finalised (no breaking changes after Phase 2)
- [ ] All component owners identified
- [ ] Monitoring metrics list agreed (see Phase 5)

**Sign-off required before Phase 1 begins.**
