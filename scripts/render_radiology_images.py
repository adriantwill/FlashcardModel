import json
from pathlib import Path

import pymupdf

seen_pdf = set()
path = Path("data/radiology_manifest.jsonl")
with path.open("r", encoding="utf-8") as f:
    for question in f:
        question = json.loads(question)
        u_id = question.get("upload_id")
        png_name = f"{u_id}_p_{question['page_number']}"
        if u_id in seen_pdf:
            continue
        doc = pymupdf.open(question["pdf_path"])
        page = doc[int(question["page_number"]) - 1]
        pixmap = page.get_pixmap(dpi=150)
        pixmap.save(png_name)
