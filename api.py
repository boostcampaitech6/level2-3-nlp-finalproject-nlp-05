from fastapi import APIRouter, HTTPException, status
from schemas import SentenceGenerationRequest, SentenceGenerationResponse, PoemImageGenerationRequest, PoemImageGenerationResponse
from dependency import get_sentence_generator, get_poem_generator
from database import GeneratedSentence, GeneratedPoemAndImage, engine
from sqlmodel import Session, select

import random
import os


router = APIRouter()


@router.post('/main')
def generate_sentences(request: SentenceGenerationRequest) -> SentenceGenerationResponse:
    selected_word = request.input_word
    
    # 문장 생성 모델 정의(아마 여기서 selected_word랑 random seed가 사용됨)
    sentence_generator = get_sentence_generator()

    # 생성된 문장들을 저장할 빈 리스트
    generated_sentences = []

    for i in range(3):
        seed = random.randint(0, 1024)
        generated_sentence = f'dummy_sentence{i+1}' # TODO : 문장을 생성
        generated_sentences.append( generated_sentence )

    result = GeneratedSentence(adj_word=selected_word,
                               generated_sentence1=generated_sentences[0],
                               generated_sentence2=generated_sentences[1],
                               generated_sentence3=generated_sentences[2])
    
    with Session( engine ) as session:
        session.add( result )
        session.commit()
        session.refresh( result )

    return SentenceGenerationResponse(id=result.id,
                                      result_sentence1=generated_sentences[0],
                                      result_sentence2=generated_sentences[1],
                                      result_sentence3=generated_sentences[2])

# 생성된 3개의 문장을 요청(GET/Selection Page)
@router.get('/line/{id}')
def get_sentences(id: int) -> SentenceGenerationResponse:
    with Session( engine ) as session:
        result = session.get(GeneratedSentence, id)
        if not result:
            raise HTTPException(detail='Not Found',
                                status_code=status.HTTP_404_NOT_FOUND)
        return SentenceGenerationResponse(id=result.id,
                                          result_sentence1=result.generated_sentence1,
                                          result_sentence2=result.generated_sentence2,
                                          result_sentence3=result.generated_sentence3)


# 선택된 문장 하나로 시 생성을 요청(POST/Selection Page)
@router.post('/line')
def generate_poem(request: PoemImageGenerationRequest, key_sentence: str) -> PoemImageGenerationResponse:
    selected_sentence = key_sentence # TODO : 문장 선택하면 그 문장으로 선택되고 POST요청되게 하기
    
    # 시 생성 모델 정의(아마 여기서 seleted_sentence가 사용됨)
    poem_generator = get_poem_generator()

    # 시 생성
    generated_poem = 'dummy poem' # TODO : 모델 구조에 따라 정의 다를듯
    
    # 이미지 생성
    generated_image = None          # TODO : DALL.E. api 적용시키기
                                    # TODO : 이미지를 저장하기
    image_path = 'dummy image path' # TODO : 저장한 이미지 경로

    # 생성된 결과를 DB에 저장
    result = GeneratedPoemAndImage(generated_poem=generated_poem,
                                   generated_image_path=image_path)
    
    with Session( engine ) as session:
        session.add( result )
        session.commit()
        session.refresh( result )

    # TODO : Redirect to 'OUTPUT PAGE'

    return PoemImageGenerationResponse(id=result.id,
                                       result_poem=result.generated_poem,
                                       result_image_path=result.generated_image_path)


# 생성된 이미지와 시를 띄워주는 웹페이지 요청(GET/Output Page)
@router.get('/poem/{id}')
def get_poem_and_image(id: int) -> PoemImageGenerationResponse:
    with Session( engine ) as session:
        result = session.get(GeneratedPoemAndImage, id)
        if not result:
            raise HTTPException(detail='Not Found',
                                status_code=status.HTTP_404_NOT_FOUND)
        return PoemImageGenerationResponse(id=result.id,
                                           result_poem=result.generated_poem,
                                           result_image_path=result.generated_image_path)