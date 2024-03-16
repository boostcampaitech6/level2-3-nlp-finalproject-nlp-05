from pydantic import BaseModel, Field
from typing import Optional

class LineRequest(BaseModel):
    emotion: str

class LineResponse(BaseModel):
    

class PoemRequest(BaseModel):
    line: str
