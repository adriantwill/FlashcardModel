from pathlib import Path

import pandas as pd
import pymupdf

if __name__ == "__main__":
    question_list = pd.read_csv("data/sql/questions_rows.csv")
    storage_set = set(question_list["storage_path"])
    for f in Path("data/pdfs").iterdir():
        if f.is_file() and f.suffix == ".pdf" and f.stem in storage_set:
            with pymupdf.open(f) as document:
                for page_number, page in enumerate(document):
                    output_path = Path(f"data/images/{f.stem}+{page_number + 1}.png")
                    page.get_pixmap(dpi=150, alpha=False).save(output_path)
                    print(f"Created {output_path}")
