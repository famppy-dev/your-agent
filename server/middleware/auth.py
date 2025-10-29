from fastapi import Header, HTTPException, status

from server.models.enums import AppErrorCode


async def get_api_key(x_api_key: str = Header(None, alias="X-API-Key")) -> str:
    if x_api_key != "secret-key-2025":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": AppErrorCode.AUTH_FAILED,
                "message": "Invalid API Key",
                "details": {},
            },
        )
    return x_api_key
