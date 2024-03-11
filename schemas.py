from pydantic import BaseModel, Field
from typing import Optional

# 형용사 단어를 선택했을 때 문장생성을 위해 모델에 전달할 Request Schema
class SentenceGenerationRequest(BaseModel):
    input_word: str

# 생성된 문장을 받는 Response Schema
class SentenceGenerationResponse(BaseModel):
    result_sentence1: str
    result_sentence2: str
    result_sentence3: str

# 시를 생성하기 시작할 문장을 모델에 전달할 Request Schema
class PoemImageGenerationRequest(BaseModel):
    input_sentence: str

# 생성된 시와 이미지를 받는 Response Schema
class PoemImageGenerationResponse(BaseModel):
    result_poem: str
    result_image_url: Optional[str] = Field(default='OPEN_AI_API_KEY is not selected', description="The path to the generated image")