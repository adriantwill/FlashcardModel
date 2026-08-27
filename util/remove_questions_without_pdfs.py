from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_CSV = PROJECT_ROOT / "data/sql/uploads_rows.csv"
QUESTIONS_CSV = PROJECT_ROOT / "data/sql/questions_rows.csv"
PDF_DIRECTORY = PROJECT_ROOT / "data/pdfs"

if __name__ == "__main__":
    questions = pd.read_csv(QUESTIONS_CSV)
    uploads = pd.read_csv(UPLOADS_CSV)
    files = [item.name for item in PDF_DIRECTORY.iterdir() if item.is_file()]
    uploads["storage_path"] = uploads["storage_path"].str.replace(
        "4nypRh2sNfV73kllqTMWoG0v0Z5zxi7f/", ""
    )
    uploads = uploads[uploads["storage_path"].isin(files)]
    questions = questions[questions["upload_id"].isin(uploads["id"].tolist())]
    questions = questions.dropna(subset=["page_number"])
    questions["storage_path"] = questions["upload_id"].map(
        uploads.set_index("id")["storage_path"]
    )
    questions["storage_path"] = questions["storage_path"].str.replace(".pdf", "")
    questions.to_csv("data/sql/questions_rows.csv", index=False)
