from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_key: str
    model: str
    base_url: str
    tavily_api_key: str = ""
    # D24 验收：Prompt 可配置 —— 提示模式（react / plan_execute / reflection）
    pattern: str = "react"

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()
