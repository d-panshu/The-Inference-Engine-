# Distributed LLM Inference Engine

A production-grade LLM inference engine running on a single i5-1135G7 laptop,
simulating a distributed GPU cluster using Ray across 8 CPU cores.

**Stack:** Python · FastAPI · Ray · Ollama · vLLM · PyTorch · Prometheus · Grafana · Docker

---

## Architecture

```
Client → FastAPI Gateway → Ray Serve → InferenceWorker (×2) → Ollama + vLLM
                                                                      ↓
                                                              Llama 3 8B (Q4)
Prometheus ← scrape /metrics ← all services
Grafana    ← query           ← Prometheus
```

---

## Phase Gates (run in order)

### Phase 0 — System design

Read `docs/architecture.md`. Contains:
- High-level architecture
- Capacity plan (16 GB RAM budget)
- API contract (OpenAI-compatible)
- Sequence diagram (full request lifecycle)
- Failure modes and mitigations

**Gate:** Document reviewed. Capacity fits within 16 GB.

---

### Phase 1 — Docker environment

```bash
# Pull and start all services
docker compose up -d

# Watch until all healthy
docker compose ps

# Expected output: all 6 services Status=healthy
# ollama, ollama-pull, ray-head, ray-worker-1, ray-worker-2, gateway,
# prometheus, grafana
```

**Gate:** `docker compose ps` shows all services healthy. Visit:
- Grafana:         http://localhost:3000  (admin/admin)
- Ray dashboard:   http://localhost:8265
- Prometheus:      http://localhost:9090

---

### Phase 2 — Inference engine

```bash
# Single request — measures TTFT and throughput
python tests/load_test.py --mode single
```

**Gate:** Response in < 10s. Tokens/sec > 0.

---

### Phase 3 — Distributed Ray layer

```bash
# 4 concurrent requests — no OOM, no errors
python tests/load_test.py --mode concurrent --users 4
```

**Gate:** 0 errors. All 4 requests complete successfully.

---

### Phase 4 — API gateway

```bash
# OpenAI SDK compatibility test
pip install openai
python - <<'EOF'
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
resp = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
EOF
```

**Gate:** OpenAI SDK works with `base_url="http://localhost:8000/v1"`.

Streaming test:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3","messages":[{"role":"user","content":"Count to 5"}],"stream":true}'
```

---

### Phase 5 — Observability

```bash
# Generate some load so metrics are populated
python tests/load_test.py --mode stress --users 2 --duration 30

# Check metrics endpoint
curl http://localhost:8000/metrics | grep llm_
```

Open Grafana at http://localhost:3000 — the "LLM Inference Engine" dashboard shows:
- TTFT P50 / P99
- Tokens per second
- Active batch size
- Queue depth
- Memory usage
- KV cache utilisation

**Gate:** All 6 panels show live data.

---

## Key Concepts Implemented

| Concept | Where | File |
|---|---|---|
| Continuous batching | Requests share decode steps without waiting | `inference/batching.py` |
| Paged attention | KV cache in fixed pages, no fragmentation | `inference/engine.py` `KVCacheTracker` |
| Tensor parallelism | Model layers split across Ray actors | `ray_cluster/serve_deployment.py` |
| OpenAI compatibility | Drop-in API replacement | `api/main.py`, `api/schemas.py` |
| Rate limiting | Token bucket per IP | `api/main.py` `@limiter.limit` |
| Structured logging | JSON logs with trace IDs | `api/main.py` request_id propagation |
| Prometheus metrics | 5 core inference metrics | `monitoring/metrics.py` |
| Alert rules | OOM, queue full, high TTFT | `monitoring/alerts.yml` |

---

## Repo Structure

```
distributed-llm-inference/
├── docs/
│   └── architecture.md          ← Phase 0: system design
├── docker/
│   ├── Dockerfile.ray
│   └── Dockerfile.gateway
├── docker-compose.yml            ← Phase 1: all services
├── inference/
│   ├── engine.py                 ← Phase 2: Ollama wrapper + KV cache
│   └── batching.py              ← Phase 2: continuous batching
├── ray_cluster/
│   └── serve_deployment.py      ← Phase 3: Ray actors + Ray Serve
├── api/
│   ├── main.py                  ← Phase 4: FastAPI + rate limiting
│   └── schemas.py               ← Phase 4: OpenAI-compatible schemas
├── monitoring/
│   ├── metrics.py               ← Phase 5: Prometheus instrumentation
│   ├── prometheus.yml
│   ├── alerts.yml
│   └── grafana/dashboards/
│       └── inference.json
└── tests/
    └── load_test.py             ← Phase gates: single/concurrent/stress
```
# The-Inference-Engine-
