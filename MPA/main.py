from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from dependencies import load_model_tokenizer, get_model_tokenizer, load_poem_model_tokenizer, get_poem_model_tokenizer
from config import config
from loguru import logger
from pydantic import BaseModel
import uvicorn

import time

import torch
import random


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # 모델&토큰나이저 로드
    load_model_tokenizer(config.model_path)
    load_poem_model_tokenizer(config.poem_model_path)
    logger.info("Loading model")
    yield

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="template") # HTML 파일이 위치한 디렉토리를 지정
app.mount("/static", StaticFiles(directory="template/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_first_page(request: Request):
    return templates.TemplateResponse("keyword_page.html", {"request": request})

@app.post("/submit_keyword")
async def handle_form(mood: str = Form(...)):
    # 사용자가 선택한 값을 다음 페이지로 전달
    return RedirectResponse(url=f"/submit_keyword/sentence?mood={mood}", status_code=303)

@app.get("/submit_keyword/sentence", response_class=HTMLResponse)
async def show_result(request: Request, mood: str):
    # 두 번째 페이지에서 사용자의 선택을 표시
    
    # 모델&토크나이저 load
    model, tokenizer = get_model_tokenizer()
    
    # 코드 실행 전 시간 측정
    start_time = time.time()
    
    # input 데이터 전처리
    inputs = tokenizer(mood, return_tensors="pt")
    # # 예측
    # outputs = model.generate(
    #     **inputs,
    #     num_beams=6,
    #     num_return_sequences=3,  # 반환할 시퀀스 수
    #     do_sample=True
    # )
    # #결과 디코딩
    # sentence = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    sentence =[]
    for i in range(3):
        output = model.generate(**inputs, do_sample=True)
        sentence.append(tokenizer.decode(output[0], skip_special_tokens=True))
    
    # 코드 실행 후 시간 측정
    end_time = time.time()

    # 실행 시간 계산 (초 단위)
    execution_time = end_time - start_time
    
    logger.info("predict time : {}", execution_time)
    # logger.info("sentence : {}", sentence)
    
    return templates.TemplateResponse("sentence_page.html", {"request": request, "mood": mood, 'sentence': sentence})

@app.post("/submit_sentence")
async def handle_form(sentence: str = Form(...)):
    # 사용자가 선택한 값을 다음 페이지로 전달
    logger.info("sentence : {}", sentence)
    return RedirectResponse(url=f"/submit_sentence/poem?sentence={sentence}", status_code=303)

@app.get("/submit_sentence/poem", response_class=HTMLResponse)
async def show_result(request: Request, sentence: str):
    # 두 번째 페이지에서 사용자의 선택을 표시
    
        # 모델&토크나이저 load
    model, tokenizer = get_poem_model_tokenizer()
    
    # 코드 실행 전 시간 측정
    start_time = time.time()
    
    # 시 생성 
    input_ids = tokenizer.encode(sentence, add_special_tokens=True, return_tensors='pt')
    output = model.generate(
        input_ids=input_ids,
        temperature=0.2, # 생성 다양성 조절
        max_new_tokens=64, # 생성되는 문장의 최대 길이
        top_k=25, # 높은 확률을 가진 top-k 토큰만 고려
        top_p=0.95, # 누적 확률이 p를 초과하는 토큰은 제외
        repetition_penalty=1.2, # 반복을 줄이는 패널티
        do_sample=True, # 샘플링 기반 생성 활성화
        num_return_sequences=1, # 생성할 시퀀스의 수
    )
    
    poem = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    
    # 코드 실행 후 시간 측정
    end_time = time.time()

    # 실행 시간 계산 (초 단위)
    execution_time = end_time - start_time
    
    logger.info("poem predict time : {}", execution_time)
    # logger.info("poem : {}", poem)
    
    return templates.TemplateResponse("output_page.html", {"request": request, "sentence": sentence, "poem": poem})

class MoodRequest(BaseModel):
    mood: str
    
@app.post("/reGenerate")
async def reGenerate(request: MoodRequest):
    
    random_seed = random.randint(1, 10000)
    torch.manual_seed(random_seed)
    
    mood = request.mood
    
    # 모델&토크나이저 load
    model, tokenizer = get_model_tokenizer()
    
    # input 데이터 전처리
    inputs = tokenizer(mood, return_tensors="pt")
    
    # 예측
    sentence =[]
    for i in range(3):
        output = model.generate(**inputs, do_sample=True)
        sentence.append(tokenizer.decode(output[0], skip_special_tokens=True))
    
    return {"sentence": sentence}


if __name__=='__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
