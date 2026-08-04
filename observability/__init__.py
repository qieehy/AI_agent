"""可观测性模块 — 日志 / trace_id / 后续接入 Langfuse。

D8 只做结构化日志 + trace_id，不做分布式追踪。
"""

from __future__ import annotations

import contextvars
import sys
from pathlib import Path

from loguru import logger

# ── trace_id：协程安全的上下文变量 ──────────────────────────
# 用法：在 Runtime.run() 入口 set，下游所有 logger 调用自动带 trace_id。
# loguru 的 .configure(patcher=...) 会在每条 log 前检查 contextvars，
# 把 trace_id 注入到 extra 字典里。
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default="unknown"
)

# ── 日志格式 ────────────────────────────────────────────────
# 开发格式（stderr 彩色）：时间 | 级别 | trace_id | 模块:行号 — 消息
DEV_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[trace_id]}</cyan> | "
    "<blue>{name}:{line}</blue> — "
    "<level>{message}</level>"
)

# JSON 格式（app.jsonl）：结构化输出，方便 grep / jq / 接入 ELK
# serialize=True 让 loguru 自动把 extra 字段也写入 JSON


def setup_logging(*, level: str = "DEBUG") -> None:
    """配置 loguru：移除默认 handler，加 stderr + JSON 两个 sink。

    只会被 main() 调用一次；多次调用安全（loguru 自动去重）。
    """
    logger.remove()  # 清空默认 handler

    # 开发用：彩色终端输出
    logger.add(
        sys.stderr,
        format=DEV_FORMAT,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # 排查用：JSON 文件输出（每次 run 自动 rotate 会更好，Week 4 做）
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    logger.add(
        logs_dir / "app.jsonl",
        format="{message}",
        level="DEBUG",
        serialize=True,  # 输出 JSON 行
        rotation="10 MB",  # 10MB 自动切分
        retention="7 days",  # 保留 7 天
    )

    # 让每条 log 自动从 contextvars 注入 trace_id
    logger.configure(patcher=lambda record: record["extra"].update(
        trace_id=trace_id_var.get()
    ))


# ── 便捷函数 ────────────────────────────────────────────────

def set_trace_id(trace_id: str) -> None:
    """在 Runtime.run() 入口调用，设置当前请求的 trace_id。"""
    trace_id_var.set(trace_id)


__all__ = ["logger", "setup_logging", "set_trace_id", "trace_id_var"]
