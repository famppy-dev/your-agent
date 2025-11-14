import json
import time
from logging import getLogger

from fastapi import APIRouter, Request, status

from server.llm.param_util import extract_json_block
from server.llm.vllm import getLlm
from server.models.rag import RagGraphRequest
from server.models.response import ApiResponse

router = APIRouter(tags=["rag"])

logger = getLogger(__name__)


@router.post("/graph", response_model=ApiResponse[dict])
async def chunking(request: Request, userRequest: RagGraphRequest):
    llm = await getLlm()

    start_time = time.time()

    response = await llm.extract_entities(userRequest.input)

    try:
        json_str = extract_json_block(text=response.outputs[0].text.strip())

        response_text = json.loads(json_str)
    except Exception as e:
        logger.warning(f"extract_json_block Error {e}")
        response_text = {"entities": [], "relations": []}

    prompt_tokens = len(response.prompt_token_ids)
    completion_tokens = len(response.outputs[0].token_ids)
    elapsed = time.time() - start_time
    tps = completion_tokens / elapsed if elapsed > 0 else 0.0

    return ApiResponse.success(
        data={
            "text": response_text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tps": tps,
            },
        },
        request=request,
        status=status.HTTP_200_OK,
        request_id=request.state.request_id,
    )
