from pydantic import Field
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    model_path: str = Field(default="gunwoo723/mt5-generate-metaphor", env="MODEL_PATH")
    poem_model_path: str = Field(default="gunwoo723/kogpt-trinity-poem-generator", env="POEM_MODEL_PATH")

config = Config()