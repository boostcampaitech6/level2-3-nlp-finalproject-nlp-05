from pydantic import BaseModel


# 형용사 단어를 선택했을 때 문장생성을 위해 모델에 전달할 Request Schema
class SentenceGenerationRequest(BaseModel):
    id: int
    input_word: str

# 생성된 문장을 받는 Response Schema
class SentenceGenerationResponse(BaseModel):
    id: int
    result_sentence1: str
    result_sentence2: str
    result_sentence3: str

# 시를 생성하기 시작할 문장을 모델에 전달할 Request Schema
class PoemImageGenerationRequest(BaseModel):
    id: int
    input_sentence: str

# 생성된 시와 이미지를 받는 Response Schema
class PoemImageGenerationResponse(BaseModel):
    id: int
    result_poem: str
    result_image_path: str