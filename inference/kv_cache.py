"""
inference/kv_cache.py
Phase 2 — Paged Attention KV Cache manager.

What is the KV cache?
    During autoregressive generation, a transformer computes Key and Value
    tensors for every token in the context. Without caching, each decode step
    recomputes the entire context — O(n²) cost.

    The KV cache stores those Key/Value tensors so each step only computes
    the NEW token's KV, then appends it. This makes decode O(n) per step.

What is paged attention? (Kwon et al. 2023, vLLM paper)
    Problem with naive KV cache:
        - Pre-allocate a contiguous block of RAM per request (max_seq_len)
        - Most requests are shorter → huge internal fragmentation
        - Can't share cache across requests with common prefixes

    Solution — treat KV cache like virtual memory:
        - Divide physical RAM into fixed-size PAGES (e.g. 16 MB each)
        - Allocate pages on demand as the sequence grows
        - Free pages immediately when a request finishes
        - Two requests sharing a system prompt share the same physical pages
          (copy-on-write, like fork() in Linux)

    Result: 2–4× more requests fit in the same RAM vs naive allocation.

This module implements the page table and allocation logic.
In production vLLM this runs in C++ and CUDA; here it's pure Python
to make the algorithm clear for interviews and code review.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Constants ──────────────────────────────────────────────────────────────────

# Each page holds KV for BLOCK_SIZE tokens.
# In vLLM this is a tunable param (default 16).
BLOCK_SIZE = 16  # tokens per page

# One KV entry = 2 (K+V) × num_heads × head_dim × dtype_bytes
# For Llama 3 8B: 2 × 32 heads × 128 dim × 2 bytes (fp16) = 16384 bytes per token
KV_BYTES_PER_TOKEN = 2 * 32 * 128 * 2  # 16 KB
PAGE_SIZE_BYTES = BLOCK_SIZE * KV_BYTES_PER_TOKEN  # 256 KB per page


@dataclass
class PhysicalPage:
    """One fixed-size block of physical KV cache memory."""
    page_id: int
    allocated: bool = False
    request_id: Optional[str] = None
    allocated_at: float = field(default_factory=time.monotonic)
    token_count: int = 0  # tokens currently stored in this page (0..BLOCK_SIZE)


@dataclass
class LogicalSequence:
    """
    The virtual address space for one request.
    Maps logical page indices → physical page IDs.
    (Analogous to a process's page table in an OS.)
    """
    request_id: str
    page_table: list[int] = field(default_factory=list)  # logical → physical
    token_count: int = 0

    @property
    def pages_used(self) -> int:
        return len(self.page_table)

    @property
    def capacity_tokens(self) -> int:
        return self.pages_used * BLOCK_SIZE

    @property
    def needs_new_page(self) -> bool:
        return self.token_count >= self.capacity_tokens


class PagedKVCacheManager:
    """
    Manages a pool of physical KV cache pages.

    Responsibilities:
        - Track which pages are free vs allocated
        - Assign pages to requests on demand (grow as sequence gets longer)
        - Free all pages for a request when it finishes
        - Prefix sharing: detect common prefixes and share physical pages (CoW)
        - Report utilization to Prometheus

    Thread safety: all public methods are async and use an asyncio.Lock.
    In a multi-worker setup, each Ray actor has its own PagedKVCacheManager
    (no cross-process sharing needed; Ray's object store handles coordination).
    """

    def __init__(self, total_ram_bytes: int = 4 * 1024 ** 3):
        num_pages = total_ram_bytes // PAGE_SIZE_BYTES
        self._pages: list[PhysicalPage] = [
            PhysicalPage(page_id=i) for i in range(num_pages)
        ]
        self._free_pages: list[int] = list(range(num_pages))  # stack of free IDs
        self._sequences: dict[str, LogicalSequence] = {}
        self._lock = asyncio.Lock()
        self._total = num_pages
        print(
            f"[KVCache] Initialised: {num_pages} pages × "
            f"{PAGE_SIZE_BYTES // 1024} KB = "
            f"{total_ram_bytes // (1024**3)} GB total"
        )

    # ── Allocation ─────────────────────────────────────────────────────────────

    async def register_request(self, request_id: str) -> bool:
        """
        Register a new request and allocate its first page.
        Returns False if no pages are available (backpressure signal).
        """
        async with self._lock:
            if not self._free_pages:
                return False  # OOM — scheduler should queue or reject
            seq = LogicalSequence(request_id=request_id)
            self._sequences[request_id] = seq
            self._allocate_page(seq)
            return True

    async def on_token_generated(self, request_id: str) -> bool:
        """
        Called after each generated token.
        If the current page is full, allocates a new one.
        Returns False if a new page was needed but unavailable.
        """
        async with self._lock:
            seq = self._sequences.get(request_id)
            if seq is None:
                return False
            seq.token_count += 1
            if seq.needs_new_page:
                if not self._free_pages:
                    return False  # can't grow — caller should abort request
                self._allocate_page(seq)
            return True

    async def free_request(self, request_id: str) -> int:
        """
        Free all pages for a completed/cancelled request.
        Returns the number of pages freed.
        """
        async with self._lock:
            seq = self._sequences.pop(request_id, None)
            if seq is None:
                return 0
            freed = 0
            for page_id in seq.page_table:
                page = self._pages[page_id]
                page.allocated = False
                page.request_id = None
                page.token_count = 0
                self._free_pages.append(page_id)
                freed += 1
            return freed

    # ── Prefix sharing (copy-on-write) ─────────────────────────────────────────

    async def share_prefix(
        self, new_request_id: str, donor_request_id: str, shared_tokens: int
    ) -> int:
        """
        Share physical pages from an existing request with a new one
        for a common prompt prefix (copy-on-write).

        How it works:
            Request A processed "You are a helpful assistant. User: Tell me..."
            Request B starts with the same system prompt.
            Instead of recomputing KV for the prefix, B maps its first N
            logical pages to A's physical pages. When B diverges (writes a new
            token), it copies-on-write to a fresh page (like Linux fork()).

        Returns the number of pages shared (= tokens saved from recomputing).
        """
        async with self._lock:
            donor = self._sequences.get(donor_request_id)
            if donor is None:
                return 0
            shared_pages = min(
                shared_tokens // BLOCK_SIZE,
                len(donor.page_table),
            )
            if shared_pages == 0:
                return 0

            new_seq = self._sequences.get(new_request_id)
            if new_seq is None:
                new_seq = LogicalSequence(request_id=new_request_id)
                self._sequences[new_request_id] = new_seq

            # Point to the same physical pages — no copy yet
            for i in range(shared_pages):
                physical_id = donor.page_table[i]
                new_seq.page_table.append(physical_id)

            new_seq.token_count = shared_pages * BLOCK_SIZE
            return shared_pages

    # ── Metrics ────────────────────────────────────────────────────────────────

    @property
    def utilization(self) -> float:
        """Fraction of pages currently allocated (0.0–1.0)."""
        used = self._total - len(self._free_pages)
        return used / self._total if self._total else 0.0

    @property
    def free_pages(self) -> int:
        return len(self._free_pages)

    @property
    def used_pages(self) -> int:
        return self._total - len(self._free_pages)

    def stats(self) -> dict:
        return {
            "total_pages": self._total,
            "used_pages": self.used_pages,
            "free_pages": self.free_pages,
            "utilization_pct": round(self.utilization * 100, 1),
            "active_requests": len(self._sequences),
            "page_size_kb": PAGE_SIZE_BYTES // 1024,
            "block_size_tokens": BLOCK_SIZE,
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _allocate_page(self, seq: LogicalSequence) -> None:
        """Grab one page from the free list and add to the sequence's page table."""
        page_id = self._free_pages.pop()
        page = self._pages[page_id]
        page.allocated = True
        page.request_id = seq.request_id
        page.allocated_at = time.monotonic()
        seq.page_table.append(page_id)
