import json
import shutil
import tempfile
import time
from logging import getLogger
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from server import MAX_UPLOAD_FILE_SIZE
from server.embedding.chunking import get_chunking_process
from server.llm.param_util import extract_json_block
from server.llm.vllm import getLlm
from server.models.enums import AppErrorCode
from server.models.rag import RagGraphRequest
from server.models.response import ApiResponse

router = APIRouter(tags=["rag"])

logger = getLogger(__name__)


@router.post("/graph/embedding", response_model=ApiResponse[dict])
async def graph_embedding(request: Request, userRequest: RagGraphRequest):
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
            "text": json.dumps(response_text),
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


@router.post("/embedding", response_model=ApiResponse[dict])
async def embedding(request: Request, file: UploadFile = File(...)):
    logger.info(file)
    try:
        processor = await get_chunking_process()

        if file is not None:
            content = await file.read()
            if file.size > MAX_UPLOAD_FILE_SIZE:
                raise HTTPException(
                    400,
                    detail=f"파일이 너무 큽니다. {MAX_UPLOAD_FILE_SIZE} 이하만 허용",
                )
            logger.info(
                f"filename: {file.filename}, size: {file.size}, content_type: {file.content_type}"
            )

            temp_dir = tempfile.mkdtemp(prefix="temp_embedding_upload_")
            file_path = Path(temp_dir) / file.filename

            # To handle file duplication
            # file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_path.suffix}")

            with open(file_path, "wb") as f:
                f.write(content)

            await processor.process_chucking(temp_dir)

            shutil.rmtree(temp_dir, ignore_errors=True)

            return ApiResponse.success(
                data={},
                request=request,
                status=status.HTTP_200_OK,
                request_id=request.state.request_id,
            )
        else:
            return ApiResponse.success(
                data={},
                request=request,
                status=status.HTTP_400_BAD_REQUEST,
                request_id=request.state.request_id,
            )
    except Exception as e:
        logger.error(f"embedding Error {e}")
        return ApiResponse.fail(
            error_code=AppErrorCode.INTERNAL,
            message=repr(e),
            request=request,
            request_id=request.state.request_id,
        )
