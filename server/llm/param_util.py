import base64
import io
import re
from pathlib import Path
from typing import List, Sequence

from PIL import Image
from vllm import PromptType

from server import getLogger
from server.models.open_ai import ImagePart, Message, TextPart

logger = getLogger(__name__)


def extract_json_block(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    cleaned = text

    close_matches = list(re.finditer(r"</think(?:ing)?>", cleaned, flags=re.IGNORECASE))
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


def convert_to_llm_string(
    messages: PromptType | Sequence[PromptType] | List[Message],
) -> str:
    text_parts = []
    image_paths = []

    if isinstance(messages, str):
        return messages, []

    for msg in messages:
        for content in msg.content:
            if isinstance(content, TextPart):
                text_parts.append(content.text)
            elif isinstance(content, ImagePart):
                # image_url은 객체이므로 .url 속성 접근
                img = image_to_base64_data_uri(content.image_url.url)
                image_paths.append(img)

    return "\n".join(text_parts), image_paths


def image_to_base64_data_uri(image_path: str, max_size=896, detail="auto") -> str:
    max_short_side = max_size  # gemma3
    max_long_side = max_size
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"이미지 없음: {image_path}")

    img = Image.open(image_path)
    img = img.convert("RGB")

    width, height = img.size
    logger.info(f"image size: {width}, {height}")

    if detail == "low":
        img = img.resize((512, 512), Image.LANCZOS)
    else:
        scale = min(
            max_short_side / min(width, height),
            max_long_side / max(width, height),
            1.0,
        )

        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)

    width, height = img.size
    logger.info(f"image resize: {width}, {height}")

    return img


def image_to_base64(img: Image) -> str:
    buffered = io.BytesIO()
    format_map = {None: "PNG", "RGB": "JPEG", "RGBA": "PNG"}
    img_format = format_map.get(img.mode, "PNG")
    img.save(buffered, format=img_format)
    img_str = base64.b64encode(buffered.getvalue()).decode()  # noqa: F821
    return f"data:image/{img_format.lower()};base64,{img_str}"


def extract_image_urls_from_messages(messages):
    image_urls = []

    for message in messages:
        if "content" in message and isinstance(message["content"], list):
            for item in message["content"]:
                if isinstance(item, dict) and item.get("type") == "image":
                    url = item.get("url")
                    if url:
                        image_urls.append(url)

    return image_urls
