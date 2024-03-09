from fastapi import APIRouter, HTTPException, status
from schemas import SentenceGenerationRequest, SentenceGenerationResponse, PoemImageGenerationRequest, PoemImageGenerationResponse
from dependency import get_sentence_generator, get_poem_generator
from database import GeneratedSentence, GeneratedPoemAndImage, engine
from sqlmodel import Session, select

import os


router = APIRouter()

#
@router.post('/main')



# 생성된 3개의 문장을 선택할 수 있는 웹페이지 요청(GET/Sentence Selection Page)
@router.get('/selection')
def get_sentences() -> list[SentenceGenerationResponse]:
    with Session( engine ) as session:
        statement = select( GeneratedSentence )
        prediction_results = session.exec( statement ).all()

        # TODO : 생성된 3개의 문장을 따로따로 표시해줄 수 있게 해야 할 듯

        return [SentenceGenerationResponse(id=prediction_result.id, result=prediction_result.result) 
                for prediction_result in prediction_results]


# 선택된 문장 하나로 시 생성을 요청(POST/Sentence Selection Page)
@router.post('/selection')
def generate_poem(request: PoemImageGenerationRequest) -> PoemImageGenerationResponse:
    selected_sentence = None # TODO : 문장 선택하면 그 문장으로 선택되고 POST요청되게 하기
    
    # 시 생성 모델 정의
    poem_generator = get_poem_generator()

    # 시 생성
    generated_poem = None # TODO : 모델 구조에 따라 정의 다를듯
    
    # 이미지 생성
    generated_image = None # TODO : DALL.E. api 적용시키기
                           # TODO : 이미지를 저장하기
    image_path = None      # TODO : 저장한 이미지 경로

    # 생성된 결과를 DB에 저장
    generated_results = GeneratedPoemAndImage(generated_poem=generated_poem,
                                              generated_image_path=image_path)
    with Session( engine ) as session:
        session.add( generated_results )
        session.commit()
        session.refresh( generated_results )

    # TODO : Redirect to 'OUTPUT PAGE'

    return PoemImageGenerationResponse(result_poem=generated_results.generated_poem,
                                       result_image_path=generated_results.generated_image_path)


# 생성된 이미지와 시를 띄워주는 웹페이지 요청(GET/Output Page)
@router.get('/output')
def get_poem_and_image() -> list[PoemImageGenerationResponse]:
    pass