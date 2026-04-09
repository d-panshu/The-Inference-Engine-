"""
api/middleware.py — Phase 4 middleware stack.
Order (outermost → innermost): Timeout → ErrorHandling → RequestLogging → route
"""
from __future__ import annotations
import asyncio, time, uuid
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from monitoring.metrics import GATEWAY_ERRORS_TOTAL

log = structlog.get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Attaches trace ID, emits structured JSON logs, adds response headers."""
    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.trace_id = trace_id
        start = time.perf_counter_ns()
        bound = log.bind(trace_id=trace_id, method=request.method,
                         path=request.url.path,
                         client_ip=request.client.host if request.client else "?")
        bound.info("request_started")
        response = await call_next(request)
        ms = (time.perf_counter_ns() - start) / 1_000_000
        response.headers["X-Request-ID"] = trace_id
        response.headers["X-Response-Time-Ms"] = f"{ms:.1f}"
        bound.info("request_completed", status=response.status_code,
                   duration_ms=round(ms, 1))
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Catches unhandled exceptions → structured JSON. Never leaks tracebacks."""
    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            return await call_next(request)
        except asyncio.TimeoutError:
            GATEWAY_ERRORS_TOTAL.labels(error_type="timeout").inc()
            return JSONResponse(504, {"error": {"type": "timeout", "code": 504,
                                                "message": "Request timed out."}})
        except MemoryError:
            GATEWAY_ERRORS_TOTAL.labels(error_type="oom").inc()
            return JSONResponse(503, {"error": {"type": "out_of_memory", "code": 503,
                                                "message": "OOM — use shorter prompt."}})
        except Exception as exc:
            GATEWAY_ERRORS_TOTAL.labels(error_type="internal").inc()
            log.error("unhandled", exc=str(exc), path=request.url.path)
            return JSONResponse(500, {"error": {"type": "internal_error", "code": 500,
                                                "message": "Internal error."}})


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Hard per-request deadline. Default 120s, set via REQUEST_TIMEOUT_S."""
    def __init__(self, app, timeout_seconds: float = 120.0):
        super().__init__(app)
        self._t = timeout_seconds

    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            return await asyncio.wait_for(call_next(request), timeout=self._t)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Exceeded {self._t}s deadline")
