import asyncio
import base64
import json
import time
import uuid
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from server import getLogger
from server.embedding.chunking import get_chunking_process
from server.llm.vllm import getLlm
from server.models.open_ai import (
    ChatCompletionRequest,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
)

router = APIRouter(tags=["open-ai-compatiple"])

logger = getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=8)


@router.post("/chat/completions")
@router.post("/chat/completion")
async def chat_completions(request: Request, userRequest: ChatCompletionRequest):
    completion_id = f"YourAgent-{uuid.uuid4()}"

    llm = await getLlm()

    created = int(time.time())
    start_time = time.time()

    response = await llm.query(
        query_str=userRequest.messages,
        temperature=userRequest.temperature,
        max_token=userRequest.max_tokens,
        top_p=userRequest.top_p,
        top_k=userRequest.top_k,
        request_id=completion_id,
        stream=userRequest.stream,
    )

    prompt_tokens = len(response.prompt_token_ids)

    def build_choice(
        index: int, content: str = "", delta: str = "", finish_reason: str | None = None
    ):
        choice = {"index": index}
        if delta:
            choice["delta"] = {"content": delta}
        else:
            choice["message"] = {"role": "assistant", "content": content}
        if finish_reason is not None:
            choice["finish_reason"] = finish_reason
        return choice

    def build_payload(
        full_text: str,
        completion_tokens: int,
        finish_reason: str = "stop",
        is_chunk: bool = False,
        tps: float | None = None,
    ):
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk" if is_chunk else "chat.completion",
            "created": created if not is_chunk else int(time.time()),
            "model": userRequest.model,
            "choices": [
                build_choice(0, content=full_text, finish_reason=finish_reason)
            ],
        }
        if not is_chunk:
            payload["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            if tps is not None:
                payload["usage"]["tps"] = round(tps, 2)
        return payload

    if userRequest.stream:

        async def stream_results() -> AsyncGenerator[bytes, None]:
            full_text = ""
            async for output in response:
                new_text = output.outputs[0].text
                delta = new_text[len(full_text) :]
                full_text = new_text
                completion_tokens = len(output.outputs[0].token_ids)

                if delta:
                    chunk = build_payload(
                        full_text="", completion_tokens=0, is_chunk=True
                    )
                    chunk["choices"][0]["delta"] = {"content": delta}
                    chunk["choices"][0]["finish_reason"] = None
                    logger.info(f"chunk: {chunk}")
                    yield f"data: {json.dumps(chunk)}\n\n"

            elapsed = time.time() - start_time
            tps = completion_tokens / elapsed if elapsed > 0 else 0.0

            final_payload = build_payload(
                full_text=full_text,
                completion_tokens=completion_tokens,
                finish_reason="stop",
                tps=tps,
            )
            yield f"data: {json.dumps(final_payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    response_text = response.outputs[0].text
    completion_tokens = len(response.outputs[0].token_ids)
    elapsed = time.time() - start_time
    tps = completion_tokens / elapsed if elapsed > 0 else 0.0
    payload = build_payload(
        full_text=response_text,
        completion_tokens=completion_tokens,
        finish_reason="stop",
        tps=tps,
    )
    logger.info(f"response: {payload}")
    return JSONResponse(payload)


@router.post("/embeddings")
@router.post("/embedding")
async def embedding(request: Request, userRequest: EmbeddingRequest):
    chunking_processor = await get_chunking_process()

    texts = (
        userRequest.input if isinstance(userRequest.input, str) else userRequest.input
    )

    loop = asyncio.get_event_loop()
    embedding_datas = await loop.run_in_executor(
        None,
        lambda: chunking_processor.get_text_embedding(texts),
    )

    if userRequest.encoding_format == "base64":
        embedding_datas = [
            base64.b64encode(embedding_datas.astype(np.float32).tobytes()).decode()
        ]

    total_tokens = sum(len(t.split()) for t in texts)

    data = [EmbeddingData(embedding=embedding_datas, index=0)]

    return EmbeddingResponse(
        data=data,
        model=userRequest.model,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    )
