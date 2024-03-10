from fastapi import APIRouter
from schemas import SentenceGenerationRequest, SentenceGenerationResponse, PoemImageGenerationRequest, PoemImageGenerationResponse
from dependency import get_sentence_generator, get_poem_generator
from openai import OpenAI

import random
import os


router = APIRouter()


@router.post('/main')
def generate_sentences(request: SentenceGenerationRequest) -> SentenceGenerationResponse:
    selected_word = request.input_word
    
    # 생성된 문장들을 저장할 빈 리스트
    generated_sentences = []

    # 문장 생성 모델 정의(아마 여기서 selected_word랑 random seed가 사용됨)
    sentence_generator = get_sentence_generator()
    for i in range(3):
        random_seed = random.randint(0, 1024)
        generated_sentence = f'dummy_sentence{i+1}' # TODO : 문장을 생성
        generated_sentences.append( generated_sentence )


    return SentenceGenerationResponse(result_sentence1=generated_sentences[0],
                                      result_sentence2=generated_sentences[1],
                                      result_sentence3=generated_sentences[2])


# 선택된 문장 하나로 시 생성을 요청(POST/Selection Page)
@router.post('/line')
def generate_poem(request: PoemImageGenerationRequest) -> PoemImageGenerationResponse:
    from openai import OpenAI

    API_KEY = None
    client = OpenAI(api_key=API_KEY)
    
    selected_sentence = request.input_sentence
    
    # 시 생성 모델 정의(아마 여기서 seleted_sentence가 사용됨)
    poem_generator = get_poem_generator()

    # 시 생성
    generated_poem = 'dummy poem' # TODO : 모델 구조에 따라 정의 다를듯
    
    # 이미지 생성
    # response = client.image.generate(model='dall-e-3',
    #                                  prompt=selected_sentence,
    #                                  size='1024x1024',
    #                                  quality='standard',
    #                                  n=1)
    # generated_image = response.data[0].url
    generated_image = 'dummy image'
    
    return PoemImageGenerationResponse(result_poem=generated_poem,
                                       result_image_path=generated_image)