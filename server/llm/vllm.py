import asyncio
from collections.abc import AsyncGenerator
from typing import List, Sequence

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer
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
from server.llm.param_util import convert_to_llm_string
from server.llm.prompts import EXTRACT_PROMPT, RAG_PROMPT_TEMPLATE
from server.models.enums import AppErrorCode
from server.models.open_ai import Message
from server.models.response import ErrorDetail

load_dotenv()

logger = getLogger(__name__)


class LlmVllm:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

        args = AsyncEngineArgs(
            model=LLM_MODEL,
            # tokenizer=self.tokenizer,
            # quantization="fp8",
            # quantization="fp4",
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
            limit_mm_per_prompt={"image": 5},
            mm_processor_kwargs={"do_pan_and_scan": False},
        )
        self.llm = AsyncLLMEngine.from_engine_args(
            args, usage_context=UsageContext.API_SERVER
        )

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
        repetition_penalty=1.0,
    ) -> RequestOutput | AsyncGenerator[RequestOutput, None] | None:
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_token,
            stop=stop,
            top_p=top_p,
            top_k=top_k,
            skip_special_tokens=False,
            stop_token_ids=[self.tokenizer.eos_token_id],
            repetition_penalty=repetition_penalty,
        )

        prompt_tokenized = self.tokenizer.apply_chat_template(
            query_str,
            tokenize=True,
            add_generation_prompt=True,  # assistant 프롬프트 추가
            return_tensors="pt",
        )

        # convert_prompt, img_parts = convert_to_llm_string(query_str)

        # logger.info(f"convert_prompt: {convert_prompt}, img_parts: {img_parts}")

        # if len(img_parts) > 0:
        #     prompts = (
        #         {
        #             "prompt": convert_prompt,
        #             "multi_modal_data": {
        #                 "image": img_parts if len(img_parts) > 1 else img_parts[0]
        #             },
        #         },
        #     )
        # else:
        #     prompts = convert_prompt

        results_generator = self.llm.generate(
            prompt={
                "prompt_token_ids": prompt_tokenized[0].tolist(),
            },
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
        return await self.query(
            RAG_PROMPT_TEMPLATE.format(context_str=context_str, query_str=query_str)
            if prompt is None
            else prompt.format(context_str=context_str, query_str=query_str)
        )

    async def extract_entities(
        self, text: str | None = None, prompt: str | None = None
    ):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": EXTRACT_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text if prompt is None else prompt},
                ],
            },
        ]
        return await self.query(messages)


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
    asyncio.run(call_main())
