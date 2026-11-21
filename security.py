"""Server-only access controls for endpoints that consume paid provider APIs."""
import os
import secrets
import threading
import time
from collections import deque

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.responses import JSONResponse

bearer = HTTPBearer(auto_error=False)
_lock = threading.Lock()
_requests = deque()


def require_api_key(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    expected = os.environ.get("CITY_RAG_API_KEY", "")
    if len(expected) < 32:
        raise HTTPException(503, "Query access is not configured")
    if credentials is None or not secrets.compare_digest(credentials.credentials.encode(), expected.encode()):
        raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": "Bearer"})
    # A shared process-wide limit cannot be bypassed by spoofing proxy headers.
    with _lock:
        now = time.monotonic()
        while _requests and _requests[0] <= now - 60:
            _requests.popleft()
        if len(_requests) >= 10:
            raise HTTPException(429, "Too many requests", headers={"Retry-After": "60"})
        _requests.append(now)


class RequestSizeLimit:
    """Bound actual bytes, including chunked bodies, before JSON parsing."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > 16384:
                return await JSONResponse({"detail": "Request too large"}, status_code=413)(scope, receive, send)
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        consumed = False
        async def bounded_receive():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()
        await self.app(scope, bounded_receive, send)
