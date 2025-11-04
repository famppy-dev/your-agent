from vllm import LLM, RequestOutput, SamplingParams
import torch
from dotenv import load_dotenv
import re

from server import LLM_DTYPE, LLM_GPU_UTIL, LLM_MAX_MODEL_LEN, LLM_MODEL, getLogger

load_dotenv()

logger = getLogger(__name__)

EXTRACT_PROMPT = """
Extract entities and relations from the text. Output **only valid JSON** in the **same language as the input text**. No code blocks, no backticks, no extra text, no thinking steps.

**Rules**:
- Types: PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT, PRODUCT, TECHNOLOGY (always in English)
- name: lowercase, normalized (e.g., "Samsung Electronics" → "samsung electronics")
- description: 1–2 sentences in **input text language**. Include any dates directly in the description (e.g., "1968년에 태어났다", "founded on November 3, 2025").
- No duplicates
- Relations: source → target, type in UPPER_SNAKE_CASE (English), description in **input text language**, past tense

**Input**: 
{text}

**Output exactly this format — nothing else**:
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "..."}}
  ],
  "relations": [
    {{"source": "...", "target": "...", "type": "...", "description": "..."}}
  ]
}}
"""

RAG_PROMPT_TEMPLATE = """You are an expert knowledge assistant. Answer the question using only the provided documents below.
If the answer is not in the documents, respond with "I don't know".

IMPORTANT: Answer in the same language as the question (e.g., if the question is in Korean, answer in Korean).

### Reference Documents:
{context_str}

### Question:
{query_str}

### Answer:"""


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

    def query(self, query_str="") -> list[RequestOutput]:
        params = SamplingParams(
            temperature=0.1,
            max_tokens=LLM_MAX_MODEL_LEN,
            stop=["<|im_end|>", "</s>", "<|endoftext|>"],
        )
        return self.llm.generate(query_str, params)

    def query_rag(self, context_str: str, query_str: str):
        return self.query(
            RAG_PROMPT_TEMPLATE.format(context_str=context_str, query_str=query_str)
        )

    def extract_json_block(self, text: str) -> str:
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

    def extract_entities(self, text: str):
        response = self.query(EXTRACT_PROMPT.format(text=text))
        import json, re

        try:
            json_str = self.extract_json_block(text=response[0].outputs[0].text.strip())

            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"extract_entities Error {e}")
            return {"entities": [], "relations": []}


if __name__ == "__main__":
    vllm = LlmVllm()
    outputs = vllm.query(query_str=["삼성의 반도체 사업 구조는?"])
    logger.info(outputs[0].outputs[0].text)
