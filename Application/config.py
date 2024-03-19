from pydantic import Field
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    model_path: str = Field(default="gunwoo723/kogpt2-generate-poem", env="MODEL_PATH")
    poem_model_path: str = Field(default="gunwoo723/kogpt2-generate-poem", env="POEM_MODEL_PATH")

config = Config()