import asyncio
import json
import re
import time
from collections.abc import AsyncGenerator
from typing import List, Sequence

import torch
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from vllm import AsyncLLMEngine, PromptType, RequestOutput, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext

from server import (
    LLM_BATCH_TOKEN_LEN,
    LLM_DTYPE,
    LLM_GPU_UTIL,
    LLM_MAX_MODEL_LEN,
    LLM_MODEL,
    getLogger,
)
from server.llm.prompts import EXTRACT_PROMPT, RAG_PROMPT_TEMPLATE
from server.models.enums import AppErrorCode
from server.models.open_ai import Message
from server.models.response import ErrorDetail

load_dotenv()

logger = getLogger(__name__)


class LlmVllm:

    def __init__(self):
        args = AsyncEngineArgs(
            model=LLM_MODEL,
            # quantization="fp8",
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=LLM_GPU_UTIL,
            # disable_custom_all_reduce=True,
            dtype=LLM_DTYPE,
            kv_cache_dtype="auto",
            max_model_len=LLM_MAX_MODEL_LEN,
            max_num_batched_tokens=LLM_BATCH_TOKEN_LEN,
            max_num_seqs=4,
            block_size=16,
            enforce_eager=True,
            enable_chunked_prefill=True,
        )
        self.llm = AsyncLLMEngine.from_engine_args(
            args, usage_context=UsageContext.API_SERVER
        )

    def _extract_json_block(self, text: str) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        cleaned = text

        close_matches = list(
            re.finditer(r"</think(?:ing)?>", cleaned, flags=re.IGNORECASE)
        )
        if close_matches:
            last_close = close_matches[-1]
            cleaned = cleaned[last_close.end() :].lstrip()

        cleaned = re.sub(
            r"<think(?:ing)?>.*?</think(?:ing)?>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, cleaned, re.DOTALL)

        if match:
            json_str = match.group(1).strip()
        else:
            if cleaned.startswith("```") and cleaned.endswith("```"):
                json_str = cleaned[3:-3].strip()
            else:
                json_str = cleaned

        if not json_str:
            return ""
        return json_str

    def _convert_to_llm_string(
        self,
        messages: PromptType | Sequence[PromptType] | List[Message],
    ) -> str:
        if isinstance(messages, list):
            try:
                # 모든 요소가 Message 인지 간단히 검사
                if all(isinstance(m, Message) for m in messages):
                    parts = []
                    for msg in messages:
                        role = msg.role.capitalize()
                        if isinstance(msg.content, str):
                            parts.append(f"[{role}] {msg.content}")
                        else:
                            texts = [p.text for p in msg.content if p.type == "text"]
                            images = [p.image for p in msg.content if p.type == "image"]
                            text = " ".join(texts) if texts else ""
                            img_str = " ".join(f"[Image: {url}]" for url in images)
                            combined = f"{text} {img_str}".strip()
                            parts.append(f"[{role}] {combined}")
                    return "\n".join(parts)
            except Exception as e:
                return f"[ERROR: failed convert] {e}"

        if isinstance(messages, (str, dict, list)):
            return str(messages)
        return repr(messages)

    async def query(
        self,
        query_str: PromptType | Sequence[PromptType] | List[Message] = "",
        temperature: float = 0.1,
        max_token: int = LLM_MAX_MODEL_LEN,
        stop: list[str] | None = ["<|im_end|>", "</s>", "<|endoftext|>"],
        top_p: float = 1.0,
        top_k: int = 0,
        request_id: str = "",
        stream: bool = False,
    ) -> RequestOutput | AsyncGenerator[RequestOutput, None] | None:
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_token,
            stop=stop,
            top_p=top_p,
            top_k=top_k,
        )
        results_generator = self.llm.generate(
            prompt=self._convert_to_llm_string(query_str),
            sampling_params=params,
            request_id=request_id,
        )

        logger.info(f"results_generator: {results_generator}")

        if stream:
            return results_generator

        final_output = None
        try:
            async for request_output in results_generator:
                final_output = request_output
        except asyncio.CancelledError as e:
            logger.error(f"vllm query error: {e}")
            raise ErrorDetail(
                error_code=AppErrorCode.INTERNAL, message=repr(e), details=None
            )

        logger.info(f"final_output: {final_output}")
        return final_output

    async def query_rag(
        self, context_str: str, query_str: str, prompt: str | None = None
    ):
        """
        If you declare `prompt` directly, you need to declare two variables: `context_str` and `query_str`.
        """
        return self.query(
            RAG_PROMPT_TEMPLATE.format(context_str=context_str, query_str=query_str)
            if prompt is None
            else prompt.format(context_str=context_str, query_str=query_str)
        )

    async def extract_entities(self, text: str = None, prompt: str = None):
        response = self.query(
            prompt if prompt is not None else EXTRACT_PROMPT.format(text=text)
        )
        import json

        try:
            json_str = self._extract_json_block(
                text=response[0].outputs[0].text.strip()
            )

            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"extract_entities Error {e}")
            return {"entities": [], "relations": []}


llm: LlmVllm | None = None
async_lock = asyncio.Lock()
initialized = False


async def getLlm() -> LlmVllm:
    global llm, initialized
    if initialized:
        return llm

    async with async_lock:
        if initialized:
            return llm

    llm = LlmVllm()

    initialized = True
    return llm


async def call_main():
    vllm = await getLlm()

    result = await vllm.query(query_str=["삼성의 반도체 사업 구조는?"])
    logger.info(result[0].outputs[0].text)


if __name__ == "__main__":
    call_main()
