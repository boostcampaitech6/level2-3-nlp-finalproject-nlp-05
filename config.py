from pydantic import Field
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    sentence_generating_model_path: str = Field(default='model1_path.joblib', env='MODEL_PATH')
    poem_generating_model_path: str = Field(default='model2_path.joblib', env='MODEL_PATH')
    app_env: str = Field(default='local', env='APP_ENV')

config = Config()