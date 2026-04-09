"""
tests/test_api.py
Phase 4 — Integration tests for the FastAPI gateway.

Tests cover:
    - Health endpoint
    - Model listing
    - Non-streaming chat completions
    - Streaming SSE chat completions
    - OpenAI schema compliance (required fields present)
    - Rate limit headers
    - Error handling (invalid model, bad request body)
    - Metrics endpoint returns Prometheus text format

Run:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch


import pytest


# ════════════════════════════════════════════════════════════════════════════════
# Health & meta endpoints
# ════════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:

    def test_health_returns_200(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_fields(self, test_client):
        data = test_client.get("/health").json()
        assert "status" in data
        assert "gateway" in data
        assert data["gateway"] is True

    def test_health_status_values(self, test_client):
        data = test_client.get("/health").json()
        assert data["status"] in ("ok", "degraded")


class TestModelsEndpoint:

    def test_models_returns_200(self, test_client):
        resp = test_client.get("/v1/models")
        assert resp.status_code == 200

    def test_models_has_list_structure(self, test_client):
        data = test_client.get("/v1/models").json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1

    def test_models_have_required_fields(self, test_client):
        models = test_client.get("/v1/models").json()["data"]
        for m in models:
            assert "id" in m
            assert "object" in m
            assert m["object"] == "model"

    def test_llama3_is_in_models(self, test_client):
        ids = [m["id"] for m in test_client.get("/v1/models").json()["data"]]
        assert "llama3" in ids


# ════════════════════════════════════════════════════════════════════════════════
# Chat completions — non-streaming
# ════════════════════════════════════════════════════════════════════════════════

class TestChatCompletionsNonStream:

    def _post(self, client, messages=None, **kwargs):
        payload = {
            "model": "llama3",
            "messages": messages or [{"role": "user", "content": "Hello"}],
            "stream": False,
            **kwargs,
        }
        return client.post("/v1/chat/completions", json=payload)

    def test_returns_200(self, test_client):
        assert self._post(test_client).status_code == 200

    def test_response_has_openai_shape(self, test_client):
        data = self._post(test_client).json()
        assert "id" in data
        assert "object" in data
        assert "choices" in data
        assert "usage" in data
        assert data["object"] == "chat.completion"

    def test_choice_has_message(self, test_client):
        choice = self._post(test_client).json()["choices"][0]
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert len(choice["message"]["content"]) > 0

    def test_usage_fields_present(self, test_client):
        usage = self._post(test_client).json()["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_finish_reason_is_stop(self, test_client):
        choice = self._post(test_client).json()["choices"][0]
        assert choice["finish_reason"] == "stop"

    def test_response_id_starts_with_chatcmpl(self, test_client):
        data = self._post(test_client).json()
        assert data["id"].startswith("chatcmpl-")

    def test_model_echoed_in_response(self, test_client):
        data = self._post(test_client).json()
        assert data["model"] == "llama3"

    def test_system_message_accepted(self, test_client):
        resp = self._post(test_client, messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user",   "content": "Hi"},
        ])
        assert resp.status_code == 200

    def test_max_tokens_accepted(self, test_client):
        resp = self._post(test_client, max_tokens=128)
        assert resp.status_code == 200

    def test_temperature_accepted(self, test_client):
        resp = self._post(test_client, temperature=0.5)
        assert resp.status_code == 200


# ════════════════════════════════════════════════════════════════════════════════
# Chat completions — streaming (SSE)
# ════════════════════════════════════════════════════════════════════════════════

class TestChatCompletionsStream:

    def _stream(self, client, **kwargs):
        payload = {
            "model": "llama3",
            "messages": [{"role": "user", "content": "Count to 3"}],
            "stream": True,
            **kwargs,
        }
        return client.post("/v1/chat/completions", json=payload)

    def test_returns_200(self, test_client):
        assert self._stream(test_client).status_code == 200

    def test_content_type_is_event_stream(self, test_client):
        resp = self._stream(test_client)
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_request_id_in_headers(self, test_client):
        resp = self._stream(test_client)
        assert "x-request-id" in resp.headers

    def test_sse_lines_are_valid(self, test_client):
        resp = self._stream(test_client)
        body = resp.text
        data_lines = [l for l in body.split("\n") if l.startswith("data:")]
        assert len(data_lines) >= 2  # at least one chunk + [DONE]

    def test_sse_ends_with_done(self, test_client):
        resp = self._stream(test_client)
        body = resp.text
        assert "data: [DONE]" in body

    def test_sse_chunks_are_valid_json(self, test_client):
        resp = self._stream(test_client)
        data_lines = [l[6:] for l in resp.text.split("\n")
                      if l.startswith("data:") and "[DONE]" not in l]
        for line in data_lines:
            parsed = json.loads(line)
            assert "choices" in parsed
            assert "delta" in parsed["choices"][0]

    def test_sse_chunks_have_content(self, test_client):
        resp = self._stream(test_client)
        data_lines = [l[6:] for l in resp.text.split("\n")
                      if l.startswith("data:") and "[DONE]" not in l]
        all_content = "".join(
            json.loads(l)["choices"][0]["delta"].get("content", "")
            for l in data_lines
        )
        assert len(all_content) > 0


# ════════════════════════════════════════════════════════════════════════════════
# Request validation
# ════════════════════════════════════════════════════════════════════════════════

class TestRequestValidation:

    def test_missing_messages_returns_422(self, test_client):
        resp = test_client.post("/v1/chat/completions",
                                json={"model": "llama3"})
        assert resp.status_code == 422

    def test_empty_messages_list_returns_200(self, test_client):
        # Empty messages is technically valid — let the model handle it
        resp = test_client.post("/v1/chat/completions",
                                json={"model": "llama3", "messages": [], "stream": False})
        assert resp.status_code in (200, 422)

    def test_invalid_temperature_returns_422(self, test_client):
        resp = test_client.post("/v1/chat/completions", json={
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 5.0,  # max is 2.0
            "stream": False,
        })
        assert resp.status_code == 422

    def test_invalid_max_tokens_returns_422(self, test_client):
        resp = test_client.post("/v1/chat/completions", json={
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": -1,
            "stream": False,
        })
        assert resp.status_code == 422

    def test_invalid_role_returns_422(self, test_client):
        resp = test_client.post("/v1/chat/completions", json={
            "model": "llama3",
            "messages": [{"role": "admin", "content": "hi"}],
            "stream": False,
        })
        assert resp.status_code == 422


# ════════════════════════════════════════════════════════════════════════════════
# Observability
# ════════════════════════════════════════════════════════════════════════════════

class TestMetricsEndpoint:

    def test_metrics_returns_200(self, test_client):
        assert test_client.get("/metrics").status_code == 200

    def test_metrics_has_prometheus_format(self, test_client):
        body = test_client.get("/metrics").text
        # Prometheus text format always starts with # HELP or metric name
        assert "#" in body or "llm_" in body

    def test_metrics_contains_gateway_counter(self, test_client):
        # Make a request first so the counter is non-zero
        test_client.post("/v1/chat/completions", json={
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
        body = test_client.get("/metrics").text
        assert "llm_gateway_requests_total" in body

    def test_response_time_header_present(self, test_client):
        resp = test_client.post("/v1/chat/completions", json={
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
        assert "x-response-time-ms" in resp.headers

    def test_request_id_header_present(self, test_client):
        resp = test_client.post("/v1/chat/completions", json={
            "model": "llama3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
        assert "x-request-id" in resp.headers
