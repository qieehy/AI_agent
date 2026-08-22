"""D21: 视觉问答服务。

OpenAIVisionService 把图片字节编码为 base64 data URL，交给视觉模型分析。
openai 只在构造时 import（模块级保持轻量）；settings 复用 config，
与 llm/client.py 同一取法（api_key / base_url）。
"""
import base64
from abc import ABC, abstractmethod

from config import get_settings


class VisionService(ABC):
    """视觉服务接口：图片字节 + 问题 -> 文本答案。"""

    @abstractmethod
    def analyze_image(self, image_bytes: bytes, question: str) -> str: ...


class OpenAIVisionService(VisionService):
    """OpenAI 兼容视觉端点（默认 gpt-4o-mini，最便宜）。

    model 在请求时传给 chat.completions.create（SDK 约定），
    构造函数只建 client——与 llm/client.py 的用法一致。
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self._model = model_name
        self._settings = get_settings()
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._settings.api_key,
                base_url=self._settings.base_url,
            )
        except ImportError as e:
            raise ImportError("视觉模型导入失败: 请先安装 openai") from e

    def analyze_image(self, image_bytes: bytes, question: str) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ]
        response = self._client.chat.completions.create(
            model=self._model, messages=messages
        )
        return response.choices[0].message.content or ""
