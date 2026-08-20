"""D20: loader 测试。

用 fpdf2 现场生成迷你 PDF（tmp_path 隔离），不依赖仓库外的真实文件。
pypdf 是主依赖、fpdf2 在 dev extras，两者 CI 皆有，本文件直接运行。
注意：fpdf2 核心字体仅覆盖 ASCII，测试文本全部用英文。
"""
import pytest
from fpdf import FPDF

from rag.loader import load_pdf


def _write_pdf(tmp_path, draw) -> str:
    """draw: 对 pdf 对象逐页画内容的回调。返回生成文件的路径。"""
    pdf = FPDF()
    draw(pdf)
    path = tmp_path / "doc.pdf"
    pdf.output(str(path))
    return str(path)


def test_returns_page_number_and_text(tmp_path):
    def draw(pdf):
        pdf.add_page()
        pdf.set_font("helvetica", size=12)
        pdf.multi_cell(0, 8, "Hello from the knowledge base")

    pages = load_pdf(_write_pdf(tmp_path, draw))

    assert len(pages) == 1
    assert pages[0].page == 1
    assert "Hello from the knowledge base" in pages[0].text


def test_page_numbers_start_at_one(tmp_path):
    def draw(pdf):
        pdf.add_page()
        pdf.set_font("helvetica", size=12)
        pdf.cell(0, 8, "page one")
        pdf.add_page()
        pdf.cell(0, 8, "page two")

    pages = load_pdf(_write_pdf(tmp_path, draw))

    assert [p.page for p in pages] == [1, 2]


def test_skips_pages_without_text_layer(tmp_path):
    """空白页（模拟扫描件页）跳过，只返回有文本层的页。"""

    def draw(pdf):
        pdf.add_page()  # 空白页
        pdf.add_page()
        pdf.set_font("helvetica", size=12)
        pdf.cell(0, 8, "only this page has text")

    pages = load_pdf(_write_pdf(tmp_path, draw))

    assert len(pages) == 1
    assert pages[0].page == 2


def test_all_blank_pdf_raises(tmp_path):
    """整本扫描件：抛错带路径，防止上层静默索引 0 块。"""
    path = _write_pdf(tmp_path, lambda pdf: pdf.add_page())

    with pytest.raises(ValueError, match="No text extracted"):
        load_pdf(path)


def test_corrupt_file_raises_without_leaking_pypdf(tmp_path):
    """损坏文件：包装成 ValueError，不向调用方泄漏 pypdf 异常。"""
    bad = tmp_path / "bad.pdf"
    bad.write_text("this is definitely not a pdf")

    with pytest.raises(ValueError, match="Cannot read PDF"):
        load_pdf(str(bad))
