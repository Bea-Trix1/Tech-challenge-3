"""Métricas HTTP com rotas normalizadas e sem conteúdo dos laudos."""

from time import perf_counter

from prometheus_client import Counter, Histogram
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUESTS = Counter(
    "triagem_http_requests_total",
    "Requisições HTTP concluídas, incluindo erros.",
    ["method", "route", "status"],
)
DURATION = Histogram(
    "triagem_http_request_duration_seconds",
    "Duração HTTP em segundos, incluindo envio da resposta.",
    ["method", "route"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)


class MetricsMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] == "/metrics":
            await self.app(scope, receive, send)
            return

        started = perf_counter()
        status = 500

        async def record_status(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, record_status)
        finally:
            route = getattr(scope.get("route"), "path", "unmatched")
            method = scope["method"]
            REQUESTS.labels(method, route, str(status)).inc()
            DURATION.labels(method, route).observe(perf_counter() - started)
