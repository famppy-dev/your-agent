from fastapi import APIRouter, Request, status

from server.models.response import ApiResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=ApiResponse[dict])
async def health_check(request: Request):

    return ApiResponse.success(
        data={"health": "ok"},
        request=request,
        status=status.HTTP_200_OK,
        request_id=request.state.request_id,
    )
