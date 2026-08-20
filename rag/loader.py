from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass(frozen=True)
class PageText:
    """单页抽取结果。page 从 1 开始。"""

    page: int
    text: str


def load_pdf(path: str) -> list[PageText]:
    """解析 PDF 为逐页文本。

    - 跳过无文本层的空页（扫描件/图片页 extract_text 为空）
    - 所有页均为空抛 ValueError 带路径：防止上层静默索引出 0 个 chunk
    - 解析失败（如文件损坏）抛 ValueError，不向调用方泄漏 pypdf 异常
    """
    pdf_path = Path(path)

    try:
        reader = PdfReader(pdf_path)

    except PdfReadError as e:
        raise ValueError(
            f"Cannot read PDF: {path}"
        ) from e


    pages: list[PageText] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if not text or not text.strip():
            continue

        pages.append(PageText(text=text, page=page_num))


    if not pages:
        raise ValueError(f"No text extracted from PDF: {path}")

    return pages
