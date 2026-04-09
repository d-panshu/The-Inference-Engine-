"""
tests/conftest.py
Shared pytest fixtures for all test modules.

Fixtures provided:
    mock_engine         — InferenceEngine with Ollama calls patched out
    mock_scheduler      — ContinuousBatchScheduler using the mock engine
    test_client         — FastAPI TestClient with Ray Serve calls patched out
    sample_request      — a ready-made InferenceRequest for reuse
    sample_messages     — OpenAI-style messages list
"""

from __future__ import annotations

import asyncio
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from inference.engine import InferenceEngine, InferenceRequest, TokenChunk
from inference.batching import ContinuousBatchScheduler
from inference.kv_cache import PagedKVCacheManager


# ── Helpers ────────────────────────────────────────────────────────────────────

FAKE_TOKENS = ["The", " transformer", " architecture", " uses", " attention", "."]


async def _fake_generate(request: InferenceRequest) -> AsyncGenerator[TokenChunk, None]:
    """Yields a fixed sequence of tokens without calling Ollama."""
    for i, tok in enumerate(FAKE_TOKENS):
        is_last = i == len(FAKE_TOKENS) - 1
        yield TokenChunk(
            request_id=request.request_id,
            token=tok,
            finish_reason="stop" if is_last else None,
            ttft_ms=42.0 if i == 0 else None,
        )
        await asyncio.sleep(0)


# ── Core fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user",   "content": "What is a transformer?"},
    ]


@pytest.fixture
def sample_request(sample_messages) -> InferenceRequest:
    return InferenceRequest(
        messages=sample_messages,
        model="llama3:8b-instruct-q4_K_M",
        max_tokens=64,
        temperature=0.0,
        request_id=str(uuid.uuid4()),
    )


@pytest.fixture
def mock_engine() -> InferenceEngine:
    """
    InferenceEngine with Ollama patched out.
    generate() yields FAKE_TOKENS without any network call.
    health_check() always returns True.
    """
    engine = InferenceEngine.__new__(InferenceEngine)
    engine._client = MagicMock()
    engine._ollama_host = "http://fake:11434"
    engine.generate = _fake_generate.__get__(engine, InferenceEngine)
    engine.health_check = AsyncMock(return_value=True)
    return engine


@pytest_asyncio.fixture
async def mock_scheduler(mock_engine) -> AsyncGenerator[ContinuousBatchScheduler, None]:
    """ContinuousBatchScheduler running against the mock engine."""
    scheduler = ContinuousBatchScheduler(
        engine=mock_engine,
        max_concurrent=2,
        max_queue_depth=8,
    )
    await scheduler.start()
    yield scheduler
    await scheduler.stop()


@pytest.fixture
def kv_cache_small() -> PagedKVCacheManager:
    """Small KV cache (16 MB) for testing page allocation logic."""
    return PagedKVCacheManager(total_ram_bytes=16 * 1024 * 1024)


# ── FastAPI test client ────────────────────────────────────────────────────────

@pytest.fixture
def test_client():
    """
    FastAPI TestClient with Ray Serve HTTP calls patched to return fake tokens.
    Allows testing the full gateway request/response pipeline without Ray running.
    """
    fake_ray_response = {
        "request_id": "test-req-001",
        "worker_id": 1,
        "tokens": FAKE_TOKENS,
        "text": "".join(FAKE_TOKENS),
        "finish_reason": "stop",
        "ttft_ms": 42.0,
        "token_count": len(FAKE_TOKENS),
    }

    with patch("api.main.httpx.AsyncClient") as mock_http_class:
        mock_http = AsyncMock()
        mock_http_class.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http_class.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value=fake_ray_response)
        mock_http.post = AsyncMock(return_value=mock_response)

        from api.main import app
        app.state.http = mock_http

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


@pytest_asyncio.fixture
async def async_test_client():
    """Async HTTPX client for testing streaming endpoints."""
    fake_ray_response = {
        "tokens": FAKE_TOKENS,
        "text": "".join(FAKE_TOKENS),
        "finish_reason": "stop",
        "ttft_ms": 42.0,
    }

    with patch("api.main.httpx.AsyncClient"):
        from api.main import app
        mock_http = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value=fake_ray_response)
        mock_http.post = AsyncMock(return_value=mock_response)
        app.state.http = mock_http

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
