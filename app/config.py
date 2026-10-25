import os 
import certifi 
from dotenv import load_dotenv

# --- AUTOMATIC SSL FIX FOR CONDA/PYTHON ENVIRONMENTS ---
ssl_cert = os.environ.get('SSL_CERT_FILE')
if ssl_cert and not os.path.exists(ssl_cert):
    os.environ['SSL_CERT_FILE'] = certifi.where()
elif not ssl_cert:
    os.environ['SSL_CERT_FILE'] = certifi.where()
# --------------------------------------------------------

load_dotenv()
from pydantic_settings import BaseSettings, SettingsConfigDict




from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str = ""
    pinecone_api_key: str = ""
    tavily_api_key: str = ""

    pinecone_index: str = "medical-agent-demo"

    gemini_chat_model: str = "gemini-3.5-flash-lite"
    gemini_embed_model: str = "gemini-embedding-001"
    gemini_vision_model: str = "gemini-3.5-flash-lite"

    api_token: str = ""
    allowed_origins: str = (
        "http://localhost:8000,http://127.0.0.1:8000"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
