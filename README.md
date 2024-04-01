# 오늘의 시  
## 프로젝트 개요
오늘의 시는 사용자의 감정을 입력받아 은유적 구절을 생성하고, 생성한 구절로 어울리는 시와 이미지를 생성하는 웹 어플리케이션입니다. 이 서비스를 통해 사용자는 자신이 느끼는 감정을 문학적 표현을 통해 풍요롭게 표현할 수 있습니다. 또, SNS에 생성한 시와 이미지를 공유하여 다른 사람들과 공감대를 형성할 수 있습니다.

## 프로젝트 타임라인
<img width="750" src="https://github.com/boostcampaitech6/level2-3-nlp-finalproject-nlp-05/assets/121072239/4fecee22-074d-4711-8a44-bb3c3ba8bffc">

## 팀원 소개
- 김기호_T6013 : 문장 데이터 생성, 문장 생성 모델 학습, Frontend, Backend
- 박상기_T6057 : 문장 데이터 생성, Frontend
- 심재혁_T6093 : 문장 데이터 생성, Backend, 배포
- 김건우_T6197 : 시 데이터 크롤링 및 전처리, 시 생성 모델 학습

## 데이터 소개
### 은유적 구절
<img width="500" alt="ChatGPT" src="https://github.com/boostcampaitech6/level2-3-nlp-finalproject-nlp-05/assets/121072239/fa8f4687-85ce-463f-829d-06420f14cc77"><br>

- 한국어 교육을 위한 감정 형용사 선정과 분류 논문에서 서비스에 활용할 감정 형용사를 선정하고, ChatGPT를 활용해 이와 어울리는 은유적 구절을 생성했습니다.
- 생성한 구절 중 잘 표현한 구절을 골라 프롬프트의 Few-shot으로 활용했습니다.
- 20가지 감정 형용사에 대해 1000개의 은유적 구절로 구성된 데이터셋을 구축했습니다.

### 시
- '시 사랑 시의 백과사전' 사이트에서 데이터 크롤링을 통해 10만개의 시 데이터를 수집했습니다.
- 원하는 형태의 시를 생성하기위해, 생성한 은유적 구절의 길이 분포를 기준으로 데이터 전처리를 진행했습니다.
- 직접 데이터 검수를 통해 수집 과정에서 처리하지 못한 노이즈를 제거했습니다.

## 모델 소개
### 은유적 구절
<img width="750" src="https://github.com/boostcampaitech6/level2-3-nlp-finalproject-nlp-05/assets/121072239/f1c2c63d-768b-43e6-98a4-21e194e41c90"><br>

- 은유적 구절 생성 모델로 Encdoer-Decoder 모델인 google/mt5-large 모델을 사용했습니다.
- 모델 input 앞에 수행할 task를 나타내는 prefix를 추가하여 모델을 학습시켰습니다.
### 시
<img width="750" src="https://github.com/boostcampaitech6/level2-3-nlp-finalproject-nlp-05/assets/121072239/f1a40df7-dff8-4dc6-bf34-9c79c56f2fbc"><br>

- 시 생성 모델로 Decoder 모델인 skt/ko-gpt-trinity-1.2B-v0.5 모델을 사용했습니다.
- 연 구분 문자를 special token에 추가하여 더 명확한 연 구분을 할 수 있도록 학습시켰습니다.

## 서비스 구조
<img width="1000" alt="스크린샷 2024-04-01 오후 5 10 14" src="https://github.com/boostcampaitech6/level2-3-nlp-finalproject-nlp-05/assets/121072239/c03f4a96-8dde-4a35-88ea-068f0b3c255f">

## 서비스 이용 방법

### 1. 은유적 구절 생성
<img width="500" alt="1" src="https://github.com/boostcampaitech6/level2-3-nlp-finalproject-nlp-05/assets/121072239/61c85cbc-0ea4-42d7-837c-f9905b765fad">
<img width="500" alt="2" src="https://github.com/boostcampaitech6/level2-3-nlp-finalproject-nlp-05/assets/121072239/c1937f81-d195-4c91-b459-239a67b2ea86"><br>

- 사용자가 선택한 감정을 바탕으로 그와 어울리는 은유적 구절을 생성합니다.
- 선택할 수 있는 감정은 크게 5가지로 각 카테고리에 4개의 단어가 존재합니다.
- 총 3개의 문장을 생성하며, 사용자는 1개를 선택할 수 있습니다.
    - 기쁨 : "기쁘다", "즐겁다", "감사하다", "행복하다"
    - 설레임 : "수줍다", "부끄럽다", "쑥쓰럽다", "민망하다"
    - 슬픔 : "슬프다", "서럽다", "속상하다", "우울하다"
    - 그리움 : "그립다", "공허하다", "외롭다", "후회스럽다"
    - 불안 : "두렵다", "불안하다", "초조하다", "혼란스럽다"

### 2. 시와 이미지 생성
<img width="500" src="https://github.com/boostcampaitech6/level2-3-nlp-finalproject-nlp-05/assets/121072239/690f17c8-517b-408c-9b57-6ba08312e7f4"><br>

- 위에서 생성한 문장을 첫 행으로 하는 시를 생성합니다.
- 시를 생성할 때, 시의 구조와 전체적인 주제 흐름을 고려합니다.
- 그림은 첫 행에서 드러나는 감정과 분위기를 기반으로 생성됩니다.
- 그림체는 수채화를 사용합니다.

### 3. SNS 공유
<img width="500" src="https://github.com/boostcampaitech6/level2-3-nlp-finalproject-nlp-05/assets/121072239/ed7a1e8f-a746-4dcb-b4a5-6ea0d41c890c"><br>

- 생성한 시와 이미지를 오늘의 시 인스타그램 공식 계정(**@poemoftoday**)에 공유할 수 있습니다.
- **인스타그램에 공유하기** 버튼을 누르고, 자신의 인스타그램 ID를 입력합니다.(비공계 계정 또는 유효하지 않은 계정은 불가)
- 업로드가 완료되고 **확인** 버튼을 누르면, 업로드된 게시물을 인스타그램에서 확인할 수 있습니다.

## 실행 방법
1. 라이브러리 설치
    - 아래의 명령어를 사용해 라이브러리를 설치합니다.

            pip install -r requirements.txt

2. app.py 실행 
    - 아래의 명령어를 사용해 서버를 실행합니다.
        
            cd Application/
            python3 app.py 
