from sqlmodel import SQLModel, Field, create_engine, Relationship
from config import config
from typing import Optional
import datetime

# 생성된 문장을 저장하는 테이블
class GeneratedSentence(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    adj_word: str
    generated_sentence1: str
    generated_sentence2: str
    generated_sentence3: str
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)

# 생성된 시와 이미지를 저장하는 테이블
class GeneratedPoemAndImage(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    generated_sentence_id: int = Field(default=None, foreign_key='generatedsentence.id')
    generated_poem: str
    generated_image_path: str
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)



engine = create_engine(config.db_url)