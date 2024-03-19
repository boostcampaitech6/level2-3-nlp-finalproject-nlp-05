from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from contextlib import asynccontextmanager
from schemas import LineRequest, PoemRequest, UploadRequest
from dependency import load_model_tokenizer, get_model_tokenizer, load_poem_model_tokenizer, get_poem_model_tokenizer
from config import config
from loguru import logger

# from openai import OpenAI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델&토큰나이저 로드
    load_model_tokenizer(config.model_path)
    load_poem_model_tokenizer(config.poem_model_path)
    logger.info("Loading model")
    yield

app = FastAPI(lifespan=lifespan)
template = FileResponse('templates/index.html')
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    return template

@app.post("/api/line")
async def generate_line(request: LineRequest):
    model, tokenizer = get_model_tokenizer()
    emotion = request.emotion
    inputs = tokenizer(emotion, return_tensors="pt")
    lines = []

    for i in range(3):
        output = model.generate(**inputs, top_k=4, do_sample=True)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        lines.append(decoded_output)

    return { "lines": lines}
    
@app.post("/api/poem")
async def generate_poem(request: PoemRequest):
    model, tokenizer = get_poem_model_tokenizer()
    
    line = request.line + '\n'
    
    # 이미지 생성
    # OpenAI API_KEY 설정
    # API_KEY = None
    # client = OpenAI(api_key=API_KEY)
    # response = client.images.generate(model='dall-e-3',
    #                                  prompt=line,
    #                                  size='1024x1024',
    #                                  quality='standard',
    #                                  n=1)
    # generated_image_url = response.data[0].url
    
    # 시 생성 
    input_ids = tokenizer.encode(line, add_special_tokens=True, return_tensors='pt')
    output = model.generate(
        input_ids=input_ids,
        temperature=0.2, # 생성 다양성 조절
        max_new_tokens=128, # 생성되는 문장의 최대 길이
        top_k=25, # 높은 확률을 가진 top-k 토큰만 고려
        top_p=0.95, # 누적 확률이 p를 초과하는 토큰은 제외
        repetition_penalty=1.2, # 반복을 줄이는 패널티
        do_sample=True, # 샘플링 기반 생성 활성화
        num_return_sequences=1, # 생성할 시퀀스의 수
    )
    poem = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

    return { "poem": poem}

@app.post("/api/upload")
async def upload(request: UploadRequest):
    id = request.instagramID

    return {"id": id}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)