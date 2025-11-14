import asyncio
from logging import getLogger

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from server import VLM_DTYPE, VLM_MAX_MODEL_LEN, VLM_MODEL
from server.llm.param_util import image_to_base64_data_uri
from server.llm.prompts.img_caption import IMG_CAPTION_PROMPT

logger = getLogger(__name__)


class Vlm:

    def __init__(self):
        self.device = "cuda"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            VLM_MODEL,
            attn_implementation="flash_attention_2",
            dtype=VLM_DTYPE,
            device_map=self.device,
        )
        self.processor = AutoProcessor.from_pretrained(VLM_MODEL)

    async def describe_image(self, img_url: str):

        img = image_to_base64_data_uri(img_url)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img,
                    },
                    {"type": "text", "text": IMG_CAPTION_PROMPT},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        conv_inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(
            **conv_inputs, max_new_tokens=VLM_MAX_MODEL_LEN, use_cache=False
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return outputs[0]


vlm: Vlm | None = None
async_lock = asyncio.Lock()
initialized = False


async def getVlm() -> Vlm:
    global vlm, initialized
    if initialized:
        return vlm

    async with async_lock:
        if initialized:
            return vlm

    vlm = Vlm()

    initialized = True
    return vlm


async def call_main():
    model = await getVlm()

    result = await model.describe_image(
        img_url="/mnt/data2/your-agent/server/embedding/data/IMG_1646.jpeg"
    )

    logger.info(result)


if __name__ == "__main__":
    asyncio.run(call_main())
