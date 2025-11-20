import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from server import getLogger
from server.embedding.chunking import get_chunking_process
from server.llm.vllm import getLlm
from server.models.enums import AppErrorCode
from server.models.response import ApiResponse, ErrorDetail

from .middleware import log, timeout
from .routers import health, open_ai, rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI initializing...")

    await get_chunking_process()
    await getLlm()

    print("FastAPI initialize complete !!")
    yield

    pass


app = FastAPI(lifespan=lifespan, title="Your Agent Middleware", version="0.1.0")

logger = getLogger(__name__)

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
        except Exception as e:
            logger.error(f"http_exception_handler: {e}")
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
app.include_router(rag.router, prefix="/v1/rag")
app.include_router(open_ai.router, prefix="/v1")


async def boot():
    import uvicorn

    # uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    asyncio.run(boot())
