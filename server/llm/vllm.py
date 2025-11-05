import re

import torch
from dotenv import load_dotenv
from vllm import LLM, RequestOutput, SamplingParams

from server import LLM_DTYPE, LLM_GPU_UTIL, LLM_MAX_MODEL_LEN, LLM_MODEL, getLogger
from server.llm.prompts import EXTRACT_PROMPT, RAG_PROMPT_TEMPLATE

load_dotenv()

logger = getLogger(__name__)


class LlmVllm:

    def __init__(self):
        self.llm = LLM(
            model=LLM_MODEL,
            # quantization="fp8",
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=LLM_GPU_UTIL,
            # disable_custom_all_reduce=True,
            dtype=LLM_DTYPE,
            kv_cache_dtype="auto",
            max_model_len=LLM_MAX_MODEL_LEN,  # max 131072 (128k)
            max_num_batched_tokens=LLM_MAX_MODEL_LEN,
            max_num_seqs=4,
            block_size=16,
            enforce_eager=True,
            enable_chunked_prefill=True,
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

    def query(
        self,
        query_str="",
        temperature: float = 0.1,
        max_token: int = LLM_MAX_MODEL_LEN,
        stop: list[str] | None = ["<|im_end|>", "</s>", "<|endoftext|>"],
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> list[RequestOutput]:
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_token,
            stop=stop,
            top_p=top_p,
            top_k=top_k,
        )
        return self.llm.generate(query_str, params)

    def query_rag(self, context_str: str, query_str: str, prompt: str | None = None):
        """
        If you declare `prompt` directly, you need to declare two variables: `context_str` and `query_str`.
        """
        return self.query(
            RAG_PROMPT_TEMPLATE.format(context_str=context_str, query_str=query_str)
            if prompt is None
            else prompt.format(context_str=context_str, query_str=query_str)
        )

    def extract_entities(self, text: str = None, prompt: str = None):
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


llm = None


def getLlm() -> LlmVllm:
    return llm if llm is not None else LlmVllm()


if __name__ == "__main__":
    vllm = getLlm()

    result = vllm.query(query_str=["삼성의 반도체 사업 구조는?"])
    logger.info(result[0].outputs[0].text)
