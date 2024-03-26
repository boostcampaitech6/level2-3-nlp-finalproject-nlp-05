from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from contextlib import asynccontextmanager
from schemas import LineRequest, LineResponse, PoemRequest, PoemResponse, UploadRequest, UploadExceptionResponse
from dependency import load_model_tokenizer, get_model_tokenizer, load_poem_model_tokenizer, get_poem_model_tokenizer
from config import config
from loguru import logger
from openai import OpenAI
from omegaconf import OmegaConf
import requests
import json
import os


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
    source_prefix = "다음 감정을 나타내는 은유적 표현을 생성해줘: "
    emotion = request.emotion
    inputs = tokenizer(source_prefix + emotion, return_tensors="pt")
    lines = []

    for i in range(3):
        output = model.generate(**inputs, temperature=1.0, min_length=8, max_length=32, do_sample=True)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        lines.append(decoded_output)

    return LineResponse(lines=lines)
    

@app.post("/api/poem")
async def generate_poem(request: PoemRequest):
    model, tokenizer = get_poem_model_tokenizer()
    line = request.line + '\n'

    # 임시 이미지
    # image_url="https://via.placeholder.com/150" 
    
    # 이미지 생성
    # OpenAI API_KEY 설정
    API_KEY_1 = os.getenv('openai_kiho')
    API_KEY_2 = os.getenv('openai_sanggi')
    API_KEY_3 = os.getenv('openai_gunwoo')
    API_KEY_4 = os.getenv('openai_jaehyeok')

    API_KEY = API_KEY_1
    
    client = OpenAI(api_key=API_KEY)
    response = client.images.generate(model='dall-e-3',
                                      prompt="Create a watercolor scene for the following sentence. Consider the factors to reinforce the feelings that fit the sentence.\n\n" + line,
                                      size='1024x1024',
                                      quality='standard',
                                      n=1)
    image_url = response.data[0].url
    
    # 시 생성 
    input_ids = tokenizer.encode(line, add_special_tokens=True, return_tensors='pt')
    output = model.generate(
        input_ids=input_ids,
        temperature=0.5, # 생성 다양성 조절
        min_length=32,   # 생성되는 문장의 최소 길이
        max_length=256, # 생성되는 문장의 최대 길이
        top_k=10, # 높은 확률을 가진 top-k 토큰만 고려
        top_p=0.95, # 누적 확률이 p를 초과하는 토큰은 제외
        repetition_penalty=1.2, # 반복을 줄이는 패널티
        do_sample=True, # 샘플링 기반 생성 활성화
        early_stopping=True, # EOS token을 만나면 조기 종료
        eos_token_id=tokenizer.eos_token_id
    )

    poem = tokenizer.decode(output[0].tolist(), skip_special_tokens=False)
    poem = poem.replace("<yun> ", "\n").replace("<s> ", "").replace("</s>", "")

    return PoemResponse(poem=poem,
                        image_url=image_url)       


@app.post("/api/upload")
async def upload(request: UploadRequest):

    id = request.instagramID
    emotion = request.emotion
    poem = request.poem 
    image_url = request.image_url

    IG_user_id = tokens.facebook.IG_user_id

    # The access_token is valid until 2024.05.17. 
    access_token = tokens.facebook.access_token

    post_url = 'https://graph.facebook.com/v19.0/{}/media'.format( IG_user_id )

    # instagram 게시할 것들
    post_payload = {
        'image_url': image_url, # 이미지
        'caption': f'#오늘의시 #{emotion}\n\n[{id}님의 오늘의 시]\n\n' + poem, # 해시태그 및 기타 입력
        'user_tags': "[ { username:'"+id+"', x: 0, y: 0 } ]", # 태그될 유저 계졍(사용자)
        'access_token': access_token
    }

    post_request = requests.post(
        post_url,
        data=post_payload
    )

    result = json.loads(post_request.text)
    
    try:
        creation_id = result['id']
    except KeyError:
        # 로깅을 추가하여 문제를 진단할 수 있도록 함
        logger.error(f"KeyError: 'id' not found in the response. Response was: {result}. Image URL: {image_url}")
        # 여기서 오류를 처리하거나 적절한 HTTP 응답을 반환할 수 있습니다.
        if result['error']['message'] == 'Invalid user id':
            return UploadExceptionResponse(error=True)
        raise HTTPException(status_code=500, detail="Internal Server Error: KeyError for 'id'.")

    publish_url = 'https://graph.facebook.com/v19.0/{}/media_publish'.format( IG_user_id )

    publish_payload = {
        'creation_id': creation_id, # 생성된 컨테이너 ID
        'access_token': access_token
    }

    publish_request = requests.post(
        publish_url, 
        data=publish_payload
    )

    return UploadExceptionResponse(error=False)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
