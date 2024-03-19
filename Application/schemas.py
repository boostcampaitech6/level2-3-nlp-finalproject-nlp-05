from pydantic import BaseModel, Field
from typing import Optional

class LineRequest(BaseModel):
    emotion: Optional[str] = None

class PoemRequest(BaseModel):
    line: Optional[str] = None

class UploadRequest(BaseModel):
    instagramID: Optional[str] = None
