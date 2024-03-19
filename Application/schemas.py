from pydantic import BaseModel, Field
from typing import Optional

class LineRequest(BaseModel):
    emotion: Optional[str] = Field(default='Emotion is not selected')

class LineResponse(BaseModel):
    lines: list

class PoemRequest(BaseModel):
    line: Optional[str] = Field(default='Line is not selected')

class PoemResponse(BaseModel):
    poem: str
    image_url: Optional[str] = Field(default='OPEN_AI_API_KEY is not selected', description='Path to the generated image')

class UploadRequest(BaseModel):
    id: Optional[str] = Field(default=None, description="User's instagram ID")