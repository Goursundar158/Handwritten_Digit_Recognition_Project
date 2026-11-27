"""Build Documentation.docx from DOCUMENTATION.md and generated results.

Run from the project root:
    python -m docs.make_documentation
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MD = PROJECT_ROOT / "DOCUMENTATION.md"
OUTPUT_DOCX = PROJECT_ROOT / "Documentation.docx"

TOKEN_PATTERN = re.compile(r"(\*\*.*?\*\*|`[^`]+`)")


def clean_inline(text: str) -> str:
    """Remove simple Markdown markers for word counting."""
    text = text.replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", text).strip()


def section_text(markdown: str, heading: str, next_heading: str) -> str:
    """Extract plain text between two Markdown headings."""
    start = markdown.index(heading) + len(heading)
    end = markdown.index(next_heading, start)
    return clean_inline(markdown[start:end])


def add_formatted_paragraph(document: Document, text: str, style: str | None = None) -> None:
    """Add a paragraph, converting **bold** and `code` markers to runs."""
    paragraph = document.add_paragraph(style=style)
    for token in TOKEN_PATTERN.split(text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            paragraph.add_run(token[2:-2]).bold = True
        elif token.startswith("`") and token.endswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        else:
            paragraph.add_run(token)


def add_markdown_table(document: Document, rows: list[list[str]]) -> None:
    """Add a Markdown table as a formatted Word table."""
    if not rows:
        return
    table = document.add_table(rows=1, cols=len(rows[0]))
    table.style = "Table Grid"
    header_cells = table.rows[0].cells
    for index, value in enumerate(rows[0]):
        header_cells[index].text = ""
        run = header_cells[index].paragraphs[0].add_run(value)
        run.bold = True
    for row in rows[1:]:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cells[index].text = value
    document.add_paragraph()


def is_separator_row(row: list[str]) -> bool:
    """Return True when a Markdown row is the |---|---| separator."""
    return all(re.fullmatch(r":?-{2,}:?", cell.strip()) for cell in row)


def parse_table_row(line: str) -> list[str]:
    """Split one Markdown table row into cell strings."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def add_footer(document: Document) -> None:
    """Add a small footer to every section."""
    for section in document.sections:
        paragraph = section.footer.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run("Handwritten Digit Recognition - Project Documentation")
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(110, 110, 110)


def build() -> None:
    """Create the Word document."""
    markdown = SOURCE_MD.read_text(encoding="utf-8")

    abstract = section_text(markdown, "## Abstract", "## Keywords")
    conclusion = section_text(markdown, "## 6. Conclusion", "## References")
    abstract_words = len(abstract.split())
    conclusion_words = len(conclusion.split())

    assert abstract_words < 200, f"Abstract has {abstract_words} words; limit is 200."
    assert conclusion_words == 150, f"Conclusion has {conclusion_words} words; expected 150."

    document = Document()
    document.core_properties.title = "Handwritten Digit Recognition"
    document.core_properties.subject = "Image Processing Project Documentation"
    document.core_properties.author = "Image Processing Project"

    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)

    # Cover page.
    for _ in range(3):
        document.add_paragraph()
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Handwritten Digit and Character Recognition System")
    run.bold = True
    run.font.size = Pt(24)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Digit Segmentation and Classification Prototype")
    run.font.size = Pt(15)

    document.add_paragraph()
    info = document.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.add_run("Hands-on Image Processing Project\n").bold = True
    info.add_run("Date: 27 November 2026\n")
    info.add_run(f"Abstract: {abstract_words} words | Conclusion: {conclusion_words} words")
    document.add_page_break()

    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if not stripped or stripped.startswith("# "):
            index += 1
            continue

        if stripped.startswith("### "):
            document.add_heading(stripped[4:], level=2)
            index += 1
            continue

        if stripped.startswith("## "):
            document.add_heading(stripped[3:], level=1)
            index += 1
            continue

        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if image_match:
            caption, relative_path = image_match.groups()
            image_path = PROJECT_ROOT / relative_path
            if image_path.exists():
                document.add_picture(str(image_path), width=Inches(6.2))
                caption_paragraph = document.add_paragraph()
                caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                caption_run = caption_paragraph.add_run(caption)
                caption_run.italic = True
                caption_run.font.size = Pt(9)
            index += 1
            continue

        if stripped.startswith("|"):
            table_rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row = parse_table_row(lines[index])
                if not is_separator_row(row):
                    table_rows.append(row)
                index += 1
            add_markdown_table(document, table_rows)
            continue

        if stripped.startswith("- "):
            add_formatted_paragraph(document, stripped, style="List Bullet")
            index += 1
            continue

        add_formatted_paragraph(document, stripped)
        index += 1

    add_footer(document)
    OUTPUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT_DOCX)
    print(f"Saved {OUTPUT_DOCX}")
    print(f"Abstract words: {abstract_words}; Conclusion words: {conclusion_words}")


if __name__ == "__main__":
    build()
