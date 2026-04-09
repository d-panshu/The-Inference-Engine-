"""
tests/test_inference.py
Phase 2 — Unit tests for the inference engine, continuous batching scheduler,
and paged KV cache manager.

Run:
    pytest tests/test_inference.py -v
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from inference.engine import InferenceEngine, InferenceRequest, TokenChunk, KVCacheTracker
from inference.batching import ContinuousBatchScheduler, QueueFullError
from inference.kv_cache import PagedKVCacheManager, BLOCK_SIZE


# ════════════════════════════════════════════════════════════════════════════════
# InferenceRequest
# ════════════════════════════════════════════════════════════════════════════════

class TestInferenceRequest:

    def test_default_request_id_is_uuid(self):
        req = InferenceRequest(messages=[{"role": "user", "content": "hi"}])
        assert len(req.request_id) == 36  # UUID format
        assert req.request_id.count("-") == 4

    def test_two_requests_have_different_ids(self):
        msgs = [{"role": "user", "content": "hi"}]
        r1 = InferenceRequest(messages=msgs)
        r2 = InferenceRequest(messages=msgs)
        assert r1.request_id != r2.request_id

    def test_defaults(self):
        req = InferenceRequest(messages=[])
        assert req.temperature == 0.7
        assert req.max_tokens == 512
        assert req.stream is True
        assert req.model == "llama3:8b-instruct-q4_K_M"


# ════════════════════════════════════════════════════════════════════════════════
# InferenceEngine (mocked Ollama)
# ════════════════════════════════════════════════════════════════════════════════

class TestInferenceEngine:

    @pytest.mark.asyncio
    async def test_generate_yields_token_chunks(self, mock_engine, sample_request):
        chunks = []
        async for chunk in mock_engine.generate(sample_request):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(isinstance(c, TokenChunk) for c in chunks)

    @pytest.mark.asyncio
    async def test_first_chunk_has_ttft(self, mock_engine, sample_request):
        chunks = []
        async for chunk in mock_engine.generate(sample_request):
            chunks.append(chunk)

        assert chunks[0].ttft_ms is not None
        assert chunks[0].ttft_ms > 0

    @pytest.mark.asyncio
    async def test_last_chunk_has_finish_reason(self, mock_engine, sample_request):
        chunks = []
        async for chunk in mock_engine.generate(sample_request):
            chunks.append(chunk)

        assert chunks[-1].finish_reason == "stop"
        # All intermediate chunks have no finish_reason
        for c in chunks[:-1]:
            assert c.finish_reason is None

    @pytest.mark.asyncio
    async def test_all_chunks_have_request_id(self, mock_engine, sample_request):
        async for chunk in mock_engine.generate(sample_request):
            assert chunk.request_id == sample_request.request_id

    @pytest.mark.asyncio
    async def test_generate_full_returns_concatenated_text(self, mock_engine, sample_request):
        text = await mock_engine.generate_full(sample_request)
        assert isinstance(text, str)
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_health_check_returns_bool(self, mock_engine):
        result = await mock_engine.health_check()
        assert isinstance(result, bool)
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_false_on_connection_error(self):
        engine = InferenceEngine(ollama_host="http://localhost:19999")
        result = await engine.health_check()
        assert result is False


# ════════════════════════════════════════════════════════════════════════════════
# KVCacheTracker (simple version in engine.py)
# ════════════════════════════════════════════════════════════════════════════════

class TestKVCacheTracker:

    @pytest.mark.asyncio
    async def test_allocate_succeeds_within_budget(self):
        tracker = KVCacheTracker(total_ram_bytes=512 * 1024 * 1024)  # 512 MB
        ok = await tracker.allocate("req-1", tokens=100)
        assert ok is True

    @pytest.mark.asyncio
    async def test_allocate_fails_when_full(self):
        # 1 MB total — barely enough for one request
        tracker = KVCacheTracker(total_ram_bytes=1 * 1024 * 1024)
        # Fill it up
        for i in range(100):
            await tracker.allocate(f"req-{i}", tokens=512)
        ok = await tracker.allocate("req-overflow", tokens=512)
        assert ok is False

    @pytest.mark.asyncio
    async def test_free_reduces_utilization(self):
        tracker = KVCacheTracker(total_ram_bytes=256 * 1024 * 1024)
        await tracker.allocate("req-a", tokens=1000)
        util_before = tracker.utilization
        await tracker.free("req-a")
        util_after = tracker.utilization
        assert util_after < util_before

    @pytest.mark.asyncio
    async def test_utilization_between_0_and_1(self):
        tracker = KVCacheTracker(total_ram_bytes=256 * 1024 * 1024)
        await tracker.allocate("req-1", tokens=500)
        assert 0.0 <= tracker.utilization <= 1.0

    @pytest.mark.asyncio
    async def test_free_unknown_request_is_safe(self):
        tracker = KVCacheTracker(total_ram_bytes=256 * 1024 * 1024)
        # Should not raise
        await tracker.free("nonexistent-req")


# ════════════════════════════════════════════════════════════════════════════════
# PagedKVCacheManager (full paged attention implementation)
# ════════════════════════════════════════════════════════════════════════════════

class TestPagedKVCacheManager:

    @pytest.fixture
    def cache(self) -> PagedKVCacheManager:
        return PagedKVCacheManager(total_ram_bytes=16 * 1024 * 1024)

    @pytest.mark.asyncio
    async def test_register_allocates_first_page(self, cache):
        ok = await cache.register_request("req-1")
        assert ok is True
        assert cache.used_pages == 1

    @pytest.mark.asyncio
    async def test_utilization_increases_after_register(self, cache):
        util_before = cache.utilization
        await cache.register_request("req-1")
        assert cache.utilization > util_before

    @pytest.mark.asyncio
    async def test_free_returns_pages(self, cache):
        await cache.register_request("req-1")
        pages_before = cache.used_pages
        freed = await cache.free_request("req-1")
        assert freed == pages_before
        assert cache.used_pages == 0

    @pytest.mark.asyncio
    async def test_token_generation_grows_pages(self, cache):
        await cache.register_request("req-1")
        pages_after_register = cache.used_pages

        # Generate BLOCK_SIZE tokens to trigger a new page allocation
        for _ in range(BLOCK_SIZE + 1):
            await cache.on_token_generated("req-1")

        assert cache.used_pages > pages_after_register

    @pytest.mark.asyncio
    async def test_multiple_requests_independent(self, cache):
        await cache.register_request("req-a")
        await cache.register_request("req-b")
        assert cache.used_pages == 2

        await cache.free_request("req-a")
        assert cache.used_pages == 1

        await cache.free_request("req-b")
        assert cache.used_pages == 0

    @pytest.mark.asyncio
    async def test_register_fails_when_no_pages(self):
        # 256 KB = exactly 1 page
        from inference.kv_cache import PAGE_SIZE_BYTES
        tiny_cache = PagedKVCacheManager(total_ram_bytes=PAGE_SIZE_BYTES)
        ok1 = await tiny_cache.register_request("req-1")
        assert ok1 is True
        ok2 = await tiny_cache.register_request("req-2")  # no pages left
        assert ok2 is False

    @pytest.mark.asyncio
    async def test_prefix_sharing_reduces_page_allocation(self, cache):
        # Register donor with a long context
        await cache.register_request("donor")
        for _ in range(BLOCK_SIZE * 3):
            await cache.on_token_generated("donor")

        pages_after_donor = cache.used_pages

        # New request shares prefix — should not allocate NEW pages
        shared = await cache.share_prefix(
            new_request_id="new-req",
            donor_request_id="donor",
            shared_tokens=BLOCK_SIZE * 2,
        )
        assert shared >= 2  # at least 2 pages shared
        # used_pages should not have jumped by 2 new allocations
        assert cache.used_pages == pages_after_donor

    @pytest.mark.asyncio
    async def test_stats_dict_has_required_keys(self, cache):
        stats = cache.stats()
        for key in ("total_pages", "used_pages", "free_pages",
                    "utilization_pct", "active_requests",
                    "page_size_kb", "block_size_tokens"):
            assert key in stats

    @pytest.mark.asyncio
    async def test_free_unknown_request_returns_zero(self, cache):
        freed = await cache.free_request("does-not-exist")
        assert freed == 0


# ════════════════════════════════════════════════════════════════════════════════
# ContinuousBatchScheduler
# ════════════════════════════════════════════════════════════════════════════════

class TestContinuousBatchScheduler:

    @pytest.mark.asyncio
    async def test_single_request_completes(self, mock_scheduler, sample_request):
        tokens = []
        async for chunk in mock_scheduler.submit(sample_request):
            tokens.append(chunk.token)
        assert len(tokens) > 0
        assert "".join(tokens)  # non-empty string

    @pytest.mark.asyncio
    async def test_concurrent_requests_all_complete(self, mock_scheduler, sample_messages):
        async def run_one(i: int) -> list[str]:
            req = InferenceRequest(
                messages=sample_messages,
                request_id=f"req-{i}",
                max_tokens=32,
            )
            tokens = []
            async for chunk in mock_scheduler.submit(req):
                tokens.append(chunk.token)
            return tokens

        # Submit 4 concurrent requests
        results = await asyncio.gather(*[run_one(i) for i in range(4)])
        assert len(results) == 4
        for tokens in results:
            assert len(tokens) > 0

    @pytest.mark.asyncio
    async def test_queue_full_raises_error(self, mock_engine):
        # Scheduler with 0 concurrent slots and 0 queue depth
        # so every submit immediately overflows
        tiny = ContinuousBatchScheduler(
            engine=mock_engine,
            max_concurrent=1,
            max_queue_depth=0,
        )
        await tiny.start()
        try:
            req = InferenceRequest(messages=[{"role": "user", "content": "x"}])
            # The queue is size 0 so put_nowait raises immediately
            with pytest.raises(QueueFullError):
                async for _ in tiny.submit(req):
                    pass
        finally:
            await tiny.stop()

    @pytest.mark.asyncio
    async def test_finish_reason_propagated(self, mock_scheduler, sample_request):
        last_chunk = None
        async for chunk in mock_scheduler.submit(sample_request):
            last_chunk = chunk
        assert last_chunk is not None
        assert last_chunk.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_ttft_only_on_first_chunk(self, mock_scheduler, sample_request):
        chunks = []
        async for chunk in mock_scheduler.submit(sample_request):
            chunks.append(chunk)

        ttft_chunks = [c for c in chunks if c.ttft_ms is not None]
        assert len(ttft_chunks) == 1
        assert ttft_chunks[0] is chunks[0]
