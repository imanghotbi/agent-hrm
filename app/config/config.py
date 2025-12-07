from pydantic import Field , SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from urllib.parse import quote_plus



class Settings(BaseSettings):
    # Required fields
    google_api_key: SecretStr
    minio_secret_key: SecretStr
    minio_access_key: str = "hrm_resume"
    structure_max_retries:int = Field(default=3)
    
    # Optional fields with defaults for minIO
    minio_endpoint: str = "http://5.75.206.1:9000"
    minio_bucket: str = "resumes"
    model_name: str = "gemini-2.5-flash"
    
    # Mongo Config
    mongo_endpoint: str
    mongo_db_name: str
    mongo_collection: str
    mongo_username: str
    mongo_password: SecretStr

    ocr_workers: int = Field(default=5, validation_alias="OCR_WORKERS")
    structure_workers: int = Field(default=10, validation_alias="STRUCTURE_WORKERS")
    eval_workers: int = Field(default=10, validation_alias="EVAL_WORKERS")

    # LLM Parameters
    max_tokens: int = 20000
    top_p: float = 0.0
    thinking_budget: int = 5000

    # logging
    log_level:str = Field(default="INFO")
    log_file_path:str = Field(default='logs/app.log')
    log_max_bytes:int = Field(default=10485760)
    log_backup_count:int = Field(default=5)

    # Configuration to load from .env file
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive=False, # Allows MINIO_ENDPOINT to map to minio_endpoint
        extra='ignore'
    )

    @property
    def mongo_uri(self):
        encoded_password = quote_plus(self.mongo_password.get_secret_value())
        return f"mongodb://{self.mongo_username}:{encoded_password}@{self.mongo_endpoint}"

# Instantiate the settings
config = Settings()
