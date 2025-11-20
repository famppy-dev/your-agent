from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrl(BaseModel):
    url: str


class ImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Union[TextPart, ImagePart]]]

    @field_validator("content", mode="before")
    def normalize_content(cls, v):
        if isinstance(v, str):
            return [{"type": "text", "text": v}]
        return v


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field("gemma3", example="gemma3")
    messages: List[Message]
    temperature: Optional[float] = Field(0.7, ge=0, le=2)
    top_p: Optional[float] = Field(1.0, ge=0, le=1)
    top_k: Optional[float] = Field(0, ge=0, le=1)
    stream: Optional[bool] = False
    is_rag: Optional[bool] = False
    max_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = Field(1.0, ge=0, le=2)


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str], List[int], List[List[int]]] = Field(
        ..., example="안녕하세요"
    )
    model: Optional[str] = Field("BAAI/bge-m3", example="BAAI/bge-m3")
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = Field(None, ge=1, le=4096)
    user: Optional[str] = None

    @field_validator("input", mode="before")
    @classmethod
    def validate_input(cls, v: Any) -> Any:
        if v is None or (isinstance(v, (str, list)) and len(v) == 0):
            raise ValueError("input must not be empty")
        return v


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: Union[List[float], str]
    index: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict
    dimensions: Optional[int] = None
