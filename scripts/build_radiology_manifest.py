import csv
import json
from pathlib import Path

rad_uploads = {}
with open("data/uploads_rows.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["folder_id"] == "e44a470c-217b-441c-90b8-90096b41e02d":
            rad_uploads[row["id"]] = row["storage_path"]
rad_questions = []
with open("data/questions_rows.csv", "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)  # radiology is e44a470c-217b-441c-90b8-90096b41e02d
    for row in reader:
        if row["upload_id"] in rad_uploads and row["page_number"]:
            pdf_path = Path(f"data/pdfs/{rad_uploads[row['upload_id']]}")
            if not pdf_path.is_file():
                pdf_path = ""
                continue
            rad_questions.append(
                {
                    "id": row["id"],
                    "upload_id": row["upload_id"],
                    "pdf_path": str(pdf_path),
                    "question": row["question_text"],
                    "answer": row["answer_text"],
                    "page_number": row["page_number"],
                }
            )
with open("data/radiology_manifest.jsonl", "w") as f:
    for question in rad_questions:
        f.write(json.dumps(question) + "\n")
