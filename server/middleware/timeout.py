import asyncio
import logging
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from server.models.enums import AppErrorCode
from server.models.response import ApiResponse

logger = logging.getLogger(__name__)


class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout_seconds: int = 10):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        try:
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            return await asyncio.wait_for(
                call_next(request), timeout=self.timeout_seconds
            )
        except TimeoutError:
            logger.warning(f"Timeout: {request.method} {request.url}")
            return ApiResponse.fail(
                error_code=AppErrorCode.TIMEOUT,
                message="Request timed out",
                request=request,
                request_id=request.state.request_id,
            )
