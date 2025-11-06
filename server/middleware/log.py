import time
import traceback

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from server import getLogger
from server.models.enums import AppErrorCode
from server.models.response import ApiResponse

logger = getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"
        logger.info(f"Request: {request.method} {request.url} from {client_ip}")

        try:

            response = await call_next(request)

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Log Exception: {tb}")
            response = ApiResponse.fail(
                error_code=AppErrorCode.INTERNAL,
                message=repr(e),
                request=request,
                request_id=request.state.request_id,
            )

        duration = time.perf_counter() - start_time
        if hasattr(response, "status_code") and hasattr(response, "request"):
            logger.info(
                f"Response: {response.status_code} {request.method} {request.url} "
                f"in {duration:.3f}s"
            )
        else:
            logger.info(f"Response: {response}" f"in {duration:.3f}s")

        return response
