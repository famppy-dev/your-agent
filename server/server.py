import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from server.models.enums import AppErrorCode
from server.models.response import ApiResponse, ErrorDetail

from .middleware import log, timeout
from .routers import health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(title="Your Agent Middleware", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modifications required during production operation
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(log.LoggingMiddleware)
app.add_middleware(timeout.TimeoutMiddleware)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):

    error_detail = None
    if isinstance(exc.detail, dict):
        try:
            error_detail = ErrorDetail(**exc.detail)
        except:
            error_detail = ErrorDetail(
                error_code=AppErrorCode.INTERNAL, message=str(exc.detail)
            )
    else:
        error_detail = ErrorDetail(
            error_code=AppErrorCode.INTERNAL, message=str(exc.detail or "Unknown error")
        )

    return ApiResponse.fail(
        error_code=error_detail.error_code,
        message=error_detail.message,
        request=request,
        details=error_detail.details,
    )


app.include_router(health.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
