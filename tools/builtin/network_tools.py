"""网络工具 — http_get（HTTP GET 请求）+ web_search（Tavily 搜索）。

安全设计：
- _safe_url() 三层沙箱防 SSRF（协议白名单 → DNS 解析 → IP 黑名单）
- http_get 所有网络异常 return 错误字符串，不抛异常（LLM 自行决定重试策略）
- 超时 / 截断 / 重定向上限 — 三项硬限制防止资源耗尽
"""

import ipaddress
import socket
from typing import Annotated
from urllib.parse import urlparse

import httpx
from tavily import TavilyClient

from config import get_settings

# ── HTTP 客户端硬限制 ──────────────────────────────────────────────
_HTTP_TIMEOUT = 10            # 请求超时（秒），防止 Agent 永久阻塞
_MAX_RESPONSE_CHARS = 10_000  # 响应体截断上限，防止 token 爆炸
_MAX_REDIRECTS = 5            # 重定向跟随上限，防止无限循环

# ── Tavily 搜索 ────────────────────────────────────────────────────
_MAX_SEARCH_RESULTS = 5       # 单次搜索返回条数上限


def _safe_url(raw: str) -> str:
    """URL 安全沙箱：三层检查防 SSRF。

    第 1 层 — 协议白名单：只放行 http / https，拒绝 file:// ftp:// gopher:// 等
    第 2 层 — DNS 解析：hostname → IP 地址
    第 3 层 — IP 黑名单：拒绝私有 / 环回 / 链路本地 / 保留地址段

    Raises:
        ValueError: URL 协议不合法、缺少 hostname 或指向内网地址
    """
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"不允许的协议: {parsed.scheme}")

    if not parsed.hostname:
        raise ValueError(f"URL 缺少 hostname: {raw}")

    info = socket.getaddrinfo(parsed.hostname, None)
    ip_str = info[0][4][0]
    ip = ipaddress.ip_address(ip_str)

    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_unspecified:
        raise ValueError(f"禁止访问内网地址: {ip_str}")

    return raw


def http_get(url: Annotated[str, "要请求的 URL（仅支持 http/https）"]) -> str:
    """发送 HTTP GET 请求，返回响应文本。

    安全特性：经过 _safe_url SSRF 沙箱校验、10 秒超时、最多 5 次重定向。
    错误处理：HTTP 错误 / 超时 / 网络异常均返回错误描述字符串，不抛异常，
    由 LLM 根据错误信息自行决定重试策略。
    内容超过 10000 字符自动截断并标注原始长度。
    """
    safe = _safe_url(url)

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT, max_redirects=_MAX_REDIRECTS) as client:
            response = client.get(safe)
            response.raise_for_status()
            text = response.text
    except httpx.HTTPStatusError as e:
        return f"HTTP 错误 [{e.response.status_code}]: {e.response.reason_phrase}"
    except httpx.TimeoutException:
        return f"请求超时 ({_HTTP_TIMEOUT}s): {url}"
    except httpx.RequestError as e:
        return f"请求失败: {type(e).__name__}: {e}"

    if len(text) > _MAX_RESPONSE_CHARS:
        text = text[:_MAX_RESPONSE_CHARS] + f"\n...truncated, original length: {len(text)}"

    return text


def web_search(query: Annotated[str, "搜索关键词"]) -> list[dict] | str:
    """用 Tavily API 搜索互联网，返回结构化结果列表。

    未配置 TAVILY_API_KEY 时返回错误提示字符串。
    返回格式：[{title, url, content, score}, ...]
    """
    key = get_settings().tavily_api_key
    if not key:
        return "error: web_search api key 未配置（请在 .env 中设置 TAVILY_API_KEY）"

    client = TavilyClient(api_key=key)
    results = client.search(
        query=query,
        max_results=_MAX_SEARCH_RESULTS,
        search_depth="basic",
    )
    return list(results["results"])


TOOLS = [http_get, web_search,]
