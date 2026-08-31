from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_key: str
    model: str
    base_url: str
    tavily_api_key: str = ""
    # D24 验收：Prompt 可配置 —— 提示模式（react / plan_execute / reflection）
    pattern: str = "react"
    # P0-1：shell_exec 执行目录由配置注入（用户无法通过命令指定）；空 = 项目根
    exec_cwd: str = ""
    tool_router_model: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        min_length=1,
    )
    tool_router_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    tool_router_top_k: int = Field(default=3, gt=0)
    tool_router_cache_size: int = Field(default=256, gt=0)
    tool_router_max_query_chars: int = Field(default=4096, gt=0)
    tool_router_worker_startup_timeout_s: float = Field(
        default=180.0,
        gt=0,
        allow_inf_nan=False,
    )
    tool_router_worker_inference_timeout_s: float = Field(
        default=30.0,
        gt=0,
        allow_inf_nan=False,
    )
    tool_router_worker_lock_timeout_s: float = Field(
        default=5.0,
        gt=0,
        allow_inf_nan=False,
    )
    tool_router_worker_shutdown_timeout_s: float = Field(
        default=5.0,
        gt=0,
        allow_inf_nan=False,
    )
    tool_router_worker_max_request_bytes: int = Field(
        default=256 * 1024,
        gt=0,
        le=8 * 1024 * 1024,
    )
    tool_router_worker_max_response_bytes: int = Field(
        default=4 * 1024 * 1024,
        gt=0,
        le=8 * 1024 * 1024,
    )
    tool_router_worker_max_stderr_chars: int = Field(
        default=4096,
        gt=0,
        le=65_536,
    )
    planner_timeout_s: float = Field(
        default=30.0,
        gt=0,
        allow_inf_nan=False,
    )
    planner_max_tasks: int = Field(
        default=12,
        gt=0,
        le=100,
    )
    planner_max_goal_chars: int = Field(
        default=1000,
        gt=0,
        le=16_384,
    )
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()
