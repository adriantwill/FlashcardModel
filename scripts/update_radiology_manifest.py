import json
from pathlib import Path

path = Path("data/radiology_manifest.jsonl")
tmp_path = path.with_suffix(".tmp")

with (
    path.open("r", encoding="utf-8") as infile,
    tmp_path.open("w", encoding="utf-8") as outfile,
):
    for line in infile:
        question = json.loads(line)
        # slide_id = f"{question['upload_id']}_p_{question['page_number']}.png"
        # question["slide_path"] = f"data/slide_images/{slide_id}"
        question.pop("page_number", None)
        question.pop("pdf_path", None)
        question.pop("upload_id", None)
        question.pop("id", None)
        outfile.write(json.dumps(question, ensure_ascii=False) + "\n")
