from fastapi import FastAPI
from contextlib import asynccontextmanager
from loguru import logger
from sqlmodel import SQLModel

from database import engine
from dependency import load_model
from config import config
from api import router


@asynccontextmanager
async def keyword_lifespan(app: FastAPI):
    # Create DB tables
    logger.info('Creating Database tables')
    SQLModel.metadata.create_all( engine )

    # Load model
    logger.info('Loading Model(Metaphor Sentence Generator)')
    load_model(config.sentence_generating_model_path)
    logger.info('Loading Model(Poem Generator)')
    load_model(config.poem_generating_model_path)

    yield

app = FastAPI(lifespan=keyword_lifespan)

