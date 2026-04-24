import csv
import json
from pathlib import Path

input_path = Path("data/radiology_manifest.jsonl")
output_path = Path("data/radiology_manifest_review.csv")

fieldnames = ["question", "answer", "slide_path", "is_good"]

with (
    input_path.open("r", encoding="utf-8") as infile,
    output_path.open("w", encoding="utf-8", newline="") as outfile,
):
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for line in infile:
        question = json.loads(line)
        slide_id = f"{question['upload_id']}_p_{question['page_number']}.png"

        writer.writerow(
            {
                "question": question["question"],
                "answer": question["answer"],
                "slide_path": f"data/slide_images/{slide_id}",
            }
        )
