import json
from pathlib import Path

import pymupdf

seen_slides = set()
path = Path("data/radiology_manifest.jsonl")
with path.open("r", encoding="utf-8") as f:
    for question in f:
        question = json.loads(question)
        upload_id = question.get("upload_id")
        png_name = f"{upload_id}_p_{question['page_number']}"
        if png_name in seen_slides:
            continue
        with pymupdf.open(question["pdf_path"]) as doc:
            page = doc[int(question["page_number"]) - 1]
            pixmap = page.get_pixmap(dpi=150)  # refactor to not reopen every time
            # pixmap.save(f"data/slide_images/{png_name}.png")
            seen_slides.add(png_name)
