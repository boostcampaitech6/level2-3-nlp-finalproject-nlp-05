from fastapi import FastAPI
from contextlib import asynccontextmanager
from loguru import logger
from sqlmodel import SQLModel

from dependency import load_sentence_generator, load_poem_generator
from config import config
from api import router


@asynccontextmanager
async def keyword_lifespan(app: FastAPI):
    # # Load model
    # logger.info('Loading Model(Metaphor Sentence Generator)')
    # load_sentence_generator(config.sentence_generating_model_path)
    # logger.info('Loading Model(Poem Generator)')
    # load_poem_generator(config.poem_generating_model_path)

    yield

app = FastAPI(lifespan=keyword_lifespan)
app.include_router( router )

@app.get('/main')
def root():
    return 'Welcome to our service'

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)