"""
api/main.py — Phase 4: FastAPI Gateway (OpenAI-compatible).
Wires middleware, rate limiting, streaming + non-streaming endpoints.
"""
from __future__ import annotations
import os, time, uuid, json, asyncio
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import httpx

from api.schemas import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionChunk, Choice, ChoiceDelta, Message, Usage,
)
from api.middleware import RequestLoggingMiddleware, ErrorHandlingMiddleware, TimeoutMiddleware
from monitoring.metrics import GATEWAY_REQUESTS_TOTAL, GATEWAY_LATENCY, GATEWAY_ERRORS_TOTAL, expose_metrics

# ── Config ─────────────────────────────────────────────────────────────────────
RAY_SERVE_URL  = os.getenv("RAY_SERVE_URL",  "http://ray-head:8001")
OLLAMA_HOST    = os.getenv("OLLAMA_HOST",    "http://ollama:11434")
RATE_LIMIT     = os.getenv("RATE_LIMIT_RPM", "60") + "/minute"
TIMEOUT_S      = float(os.getenv("REQUEST_TIMEOUT_S", "120"))

# ── App ────────────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Distributed LLM Inference Engine",
              description="OpenAI-compatible API over Ray + Ollama", version="1.0.0")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware registration (last added = outermost)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(TimeoutMiddleware, timeout_seconds=TIMEOUT_S)

# ── Lifecycle ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    app.state.http = httpx.AsyncClient(timeout=TIMEOUT_S)

@app.on_event("shutdown")
async def shutdown():
    await app.state.http.aclose()

# ── Health & meta ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        r = await app.state.http.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False
    return {"status": "ok" if ollama_ok else "degraded", "ollama": ollama_ok, "gateway": True}

@app.get("/metrics")
async def metrics():
    return expose_metrics()

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": "llama3",  "object": "model", "created": 1700000000, "owned_by": "local"},
        {"id": "mistral", "object": "model", "created": 1700000000, "owned_by": "local"},
    ]}

# ── Main inference endpoint ────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
@limiter.limit(RATE_LIMIT)
async def chat_completions(request: Request, body: ChatCompletionRequest):
    GATEWAY_REQUESTS_TOTAL.labels(model=body.model, stream=str(body.stream)).inc()
    request_id = f"chatcmpl-{uuid.uuid4().hex[:20]}"
    start = time.monotonic()

    payload = {
        "messages":    [m.model_dump() for m in body.messages],
        "model":       body.model,
        "max_tokens":  body.max_tokens or 512,
        "temperature": body.temperature or 0.7,
        "stream":      body.stream,
        "request_id":  request_id,
    }

    if body.stream:
        return StreamingResponse(
            _stream_sse(app.state.http, payload, request_id, start, body.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Request-ID": request_id},
        )
    return await _non_stream(app.state.http, payload, request_id, start, body)


async def _stream_sse(http, payload, request_id, start, model) -> AsyncGenerator[str, None]:
    created = int(time.time())
    try:
        resp = await http.post(f"{RAY_SERVE_URL}/", json=payload)
        if resp.status_code == 429:
            yield f"data: {json.dumps({'error': {'message': 'Queue full', 'code': 429}})}\n\n"
            return
        result = resp.json()
        if "error" in result:
            GATEWAY_ERRORS_TOTAL.labels(error_type=result["error"]).inc()
            yield f"data: {json.dumps({'error': result})}\n\n"
            return

        tokens: list[str] = result.get("tokens", [])
        for i, token in enumerate(tokens):
            chunk = ChatCompletionChunk(
                id=request_id, created=created, model=model,
                choices=[ChoiceDelta(index=0, delta={"content": token},
                                     finish_reason=None if i < len(tokens)-1 else "stop")],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            await asyncio.sleep(0)
        yield "data: [DONE]\n\n"
    except Exception as exc:
        GATEWAY_ERRORS_TOTAL.labels(error_type="stream_error").inc()
        yield f"data: {json.dumps({'error': {'message': str(exc), 'code': 500}})}\n\n"
    finally:
        GATEWAY_LATENCY.observe(time.monotonic() - start)


async def _non_stream(http, payload, request_id, start, body) -> ChatCompletionResponse:
    try:
        resp = await http.post(f"{RAY_SERVE_URL}/", json=payload)
        result = resp.json()
        if "error" in result:
            GATEWAY_ERRORS_TOTAL.labels(error_type=result["error"]).inc()
            raise HTTPException(status_code=result.get("status", 500),
                                detail=result.get("error"))
        text = result.get("text", "")
        prompt_tokens = sum(len(m.content.split()) for m in body.messages)
        completion_tokens = len(text.split())
        GATEWAY_LATENCY.observe(time.monotonic() - start)
        return ChatCompletionResponse(
            id=request_id, created=int(time.time()), model=body.model,
            choices=[Choice(index=0, message=Message(role="assistant", content=text),
                            finish_reason=result.get("finish_reason", "stop"))],
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
