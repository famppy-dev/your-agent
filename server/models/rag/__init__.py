from typing import Optional

from fastapi import File, UploadFile
from pydantic import BaseModel, Field


class RagGraphRequest(BaseModel):
    input: Optional[str] = Field(None, example="...")
    file: Optional[UploadFile] = File(None)
