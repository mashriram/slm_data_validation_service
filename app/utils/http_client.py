from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx

# A simple manager to reuse the client session across requests for performance.
_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def get_http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provides a single, reused httpx.AsyncClient instance."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient()

    try:
        yield _client
    finally:
        # In a real-world high-concurrency app, you might manage this lifecycle
        # with FastAPI's lifespan events for graceful shutdown.
        pass


async def close_http_client():
    """Function to be called on application shutdown."""
    global _client
    if _client:
        await _client.aclose()
        _client = None
