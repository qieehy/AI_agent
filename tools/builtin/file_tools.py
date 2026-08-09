from pathlib import Path
from typing import Annotated

_ALLOWED_EXTENSIONS: frozenset[str] | None = frozenset(
    {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".csv", ".html", ".css", ".js", ".ts",
     ".log"})
_MAX_WRITE_BYTES = 1_000_000  # 写操作 1MB 上限
_PROTECTED_FILES: frozenset[str] = frozenset({".env", ".gitignore", ".git/config", ".ssh", ".venv"})


def _safe_path(raw: str, root: Path) -> Path:
    re_root = Path(root).resolve()
    resolved = (re_root / raw).resolve()
    if not resolved.is_relative_to(re_root):
        raise ValueError(f"路径越界: {raw}")
    return resolved


_ROOTS = Path.cwd().resolve()


def read_file(raw: Annotated[str, "要读取的文件路径 (相对于项目根目录)"]) -> str:
    """读取项目根目录下文件内容"""
    safe = _safe_path(raw, _ROOTS)
    if not safe.exists():
        raise FileNotFoundError(f"文件不存在: {raw}")

    if _ALLOWED_EXTENSIONS is not None and safe.suffix not in _ALLOWED_EXTENSIONS:
        raise PermissionError(f"不允许的文件类型: {safe.suffix}")

    # 拒绝二进制文件（读前 512 字节检查是否含 null）
    with open(safe, "rb") as f:
        head = f.read(512)
    if b"\x00" in head:
        raise PermissionError(f"疑似二进制文件，拒绝读取: {raw}")

    content = safe.read_text(encoding="utf-8")
    if len(content) > 10_000:
        content = content[:10_000] + "\n...[truncated]"

    return content


def write_file(raw: Annotated[str, "要写入的文件路径 (相对于项目根目录)"], content: Annotated[str, "要写入的内容"]) -> str:
    """对在项目根目录下的文件进行写入"""
    safe = _safe_path(raw, _ROOTS)

    for protected in _PROTECTED_FILES:
        if str(safe).endswith(protected) or protected in str(safe):
            raise PermissionError(f"禁止修改受保护文件: {protected}")

    if len(content) > _MAX_WRITE_BYTES:
        raise ValueError(f"内容过大: {len(content)} > {_MAX_WRITE_BYTES} 字节")

    safe.write_text(content, encoding="utf-8")
    return f"已写入: {raw} ({len(content)}字符)"


def list_dir(path: Annotated[str, "要列出内容的目录路径（相对于项目根目录)"] )-> list[str]:
    """列出项目内目录的文件和子目录"""
    safe = _safe_path(path, _ROOTS)
    if not safe.is_dir():
        raise ValueError(f"不是目录: {path}")

    return [p.name for p in safe.iterdir()]


TOOLS = [read_file, write_file, list_dir]
