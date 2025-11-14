import asyncio
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from server import getLogger
from server.models.enums import AppErrorCode
from server.models.response import ApiResponse

logger = getLogger(__name__)


class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout_seconds: int = 180):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        try:
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            timeout = self.timeout_seconds
            return await asyncio.wait_for(call_next(request), timeout=timeout)
        except TimeoutError:
            logger.warning(f"Timeout: {request.method} {request.url}")
            return ApiResponse.fail(
                error_code=AppErrorCode.TIMEOUT,
                message="Request timed out",
                request=request,
                request_id=request.state.request_id,
            )
