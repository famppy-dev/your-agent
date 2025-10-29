import time
from typing import Any, Dict, Generic, Optional, TypeVar

from fastapi import Request, status
from pydantic import BaseModel, Field

from server.models.enums import AppErrorCode

T = TypeVar("T")


class ErrorDetail(BaseModel):
    error_code: AppErrorCode = Field(..., example=AppErrorCode.NOT_FOUND)
    message: str = Field(..., example="Not found user")
    details: Optional[Dict[str, Any]] = Field(None, example={"field": "user_name"})


class ApiResponseMeta(BaseModel):
    timestamp: float = Field(..., example=123.123)
    path: str = Field(..., example="/api/v1/user/123")
    request_id: Optional[str] = Field(None, example="req-123")


class ApiResponse(BaseModel, Generic[T]):
    """fastapi.status 기반 응답"""

    result: int = Field(..., example=200, description="HTTP status code")
    data: Optional[T] = Field(None, example={})
    error: Optional[ErrorDetail] = Field(None, example=None)
    meta: ApiResponseMeta = Field(...)

    @staticmethod
    def success(
        data: T,
        request: Request,
        status: int = status.HTTP_200_OK,
        request_id: Optional[str] = None,
    ) -> "ApiResponse[T]":
        return ApiResponse(
            result=status,
            data=data,
            meta=ApiResponseMeta(
                timestamp=time.time(),
                path=request.url.path,
                request_id=request_id,
            ),
        )

    @staticmethod
    def fail(
        error_code: AppErrorCode,
        message: str,
        request: Request,
        status: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> "ApiResponse[None]":
        return ApiResponse(
            result=status,
            data=None,
            error=ErrorDetail(error_code=error_code, message=message, details=details),
            meta=ApiResponseMeta(
                timestamp=time.time(),
                path=request.url.path,
                request_id=request_id,
            ),
        )
