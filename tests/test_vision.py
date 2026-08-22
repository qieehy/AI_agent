"""D21: VisionService 契约测试（test-first：OpenAIVisionService 与工厂落地前为红）。

策略：
- 假 openai 模块注入 sys.modules：不联网、不碰真实端点，CI 没装 openai 也能跑
- 懒加载用子进程隔离验证（与 D18/D21-3 同一纪律：模块级不碰 openai，构造时才 import）
- 只钉契约：抽象性、默认模型、base64 data URL 格式、content 透传、工厂与 __all__；
  不测真实视觉识别的语义质量（那是接真实端点后的手工验收）
"""
import base64
import builtins
import subprocess
import sys
from types import SimpleNamespace

import pytest

import rag
import rag.vision as vision_module


def _inject_fake_openai(monkeypatch, respond="看到一只猫"):
    """假 openai 模块：OpenAI(...) 记录构造参数，chat.completions.create 返回预设答案。"""
    calls = {}

    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            calls["messages"] = kwargs["messages"]
            calls["create_kwargs"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=respond))]
            )

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        def __init__(self, **kwargs):
            calls["init_kwargs"] = kwargs
            self.chat = FakeChat()

    class FakeModule:
        OpenAI = FakeClient

    monkeypatch.setitem(sys.modules, "openai", FakeModule())
    return calls


# ---------- 懒加载 ----------

def test_vision_module_never_touches_openai_at_import():
    """vision.py 模块级不得 import openai（历史教训：模块级重导入踩过两次）。

    隔离加载 vision.py 文件本身并封锁 openai 的 import：能成功导入
    = 模块级只碰 abc 等轻量依赖。不测整条 rag 链——llm/client.py
    本来就急切 import openai，与本模块无关。
    """
    code = (
        "import builtins, importlib.util\n"
        "real = builtins.__import__\n"
        "def guard(name, *a, **k):\n"
        "    if name == 'openai' or name.startswith('openai.'):\n"
        "        raise ImportError('openai blocked')\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = guard\n"
        f"spec = importlib.util.spec_from_file_location('vision_iso', {str(vision_module.__file__)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "OK"


# ---------- ABC ----------

def test_vision_service_is_abstract():
    with pytest.raises(TypeError):
        vision_module.VisionService()


def test_subclass_must_implement_analyze_image():
    class Incomplete(vision_module.VisionService):
        pass

    with pytest.raises(TypeError):
        Incomplete()


# ---------- OpenAIVisionService ----------

def test_default_model_is_gpt4o_mini(monkeypatch):
    """默认模型在请求时传给 create（SDK 约定，与 llm/client.py 一致），不是构造函数参数。"""
    calls = _inject_fake_openai(monkeypatch)

    vision_module.OpenAIVisionService().analyze_image(b"x", "是什么？")

    assert calls["create_kwargs"]["model"] == "gpt-4o-mini"


def test_image_encoded_as_base64_data_url(monkeypatch):
    """契约核心：图片字节 → base64 data URL；问题文本进 text 段，图片进 image_url 段。"""
    calls = _inject_fake_openai(monkeypatch)
    service = vision_module.OpenAIVisionService()
    image = b"\x89PNG\r\n"
    expected_url = "data:image/png;base64," + base64.b64encode(image).decode("ascii")

    service.analyze_image(image, "图里有什么？")

    messages = calls["messages"]
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": "图里有什么？"}
    assert content[1] == {"type": "image_url", "image_url": {"url": expected_url}}


def test_answer_passthrough(monkeypatch):
    _inject_fake_openai(monkeypatch, respond="一只猫趴在窗台上")

    answer = vision_module.OpenAIVisionService().analyze_image(b"x", "是什么？")

    assert answer == "一只猫趴在窗台上"


def test_content_none_returns_empty_string(monkeypatch):
    """API 安全过滤返回 content=None 时回退空串（与 pipeline.ask 同一守卫）。"""
    _inject_fake_openai(monkeypatch, respond=None)

    answer = vision_module.OpenAIVisionService().analyze_image(b"x", "是什么？")

    assert answer == ""


def test_missing_openai_raises_import_error(monkeypatch):
    """openai 未安装时构造抛 ImportError（B904 包装），而不是 AttributeError。"""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai" or name.startswith("openai."):
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        vision_module.OpenAIVisionService()


# ---------- 工厂与导出 ----------

def test_factory_returns_vision_service(monkeypatch):
    _inject_fake_openai(monkeypatch)

    service = rag.create_vision_service()

    assert isinstance(service, vision_module.VisionService)


def test_all_exports_vision_names():
    assert "VisionService" in rag.__all__
    assert "OpenAIVisionService" in rag.__all__
    assert "create_vision_service" in rag.__all__
