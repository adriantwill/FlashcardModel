from pathlib import Path

import pymupdf

if __name__ == "__main__":
    for f in Path("data/pdfs").iterdir():
        if f.is_file() and f.suffix == ".pdf":
            with pymupdf.open(f) as document:
                for page_number, page in enumerate(document, start=1):
                    output_path = Path(f"data/pdf_images/{f.stem}_{page_number}.png")
                    page.get_pixmap(dpi=150, alpha=False).save(output_path)
                    print(f"Created {output_path}")
