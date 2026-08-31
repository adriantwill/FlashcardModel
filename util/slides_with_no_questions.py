from pathlib import Path

import pandas as pd


def main():
    df = pd.read_csv("data/sql/questions_rows.csv")
    path = [path.stem for path in Path("data/images").iterdir()]
    all_images = pd.DataFrame(
        (name.split("+") for name in path), columns=["storage_path", "page_number"]
    )
    df["img_dir"] = df["storage_path"] + "+" + df["page_number"].astype(int).astype(str)
    all_images["img_dir"] = (
        all_images["storage_path"]
        + "+"
        + all_images["page_number"].astype(int).astype(str)
    )
    rows_to_add = all_images[~all_images["img_dir"].isin(df["img_dir"])]
    rows_to_add["question_text"] = "[]"
    rows_to_add["answer_text"] = "[]"
    rows_to_add = rows_to_add.drop("img_dir", axis=1)
    rows_to_add.to_csv("data/sql/empty_slides.csv", index=False)


if __name__ == "__main__":
    main()
