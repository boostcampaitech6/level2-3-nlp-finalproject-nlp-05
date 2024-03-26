from pydantic import BaseModel, Field
from typing import Optional

class LineRequest(BaseModel):
    emotion: Optional[str] = Field(default='Emotion is not selected')

class LineResponse(BaseModel):
    lines: list

class PoemRequest(BaseModel):
    line: Optional[str] = None

class PoemResponse(BaseModel):
    poem: str
    image_url: Optional[str] = Field(default='OPEN_AI_API_KEY is not selected', description='Path to the generated image')

class UploadRequest(BaseModel):
    instagramID: Optional[str] = None
    line: Optional[str] = Field(default='Line is not selected')

class UploadExceptionResponse(BaseModel):
    message: str