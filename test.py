import pandas as pd

df = pd.read_csv("data/sql/questions_rows.csv")
df = df.dropna(subset=["question_text", "answer_text"])
group_columns = ["storage_path", "page_number"]
df["chunk"] = df.groupby(group_columns, sort=False).cumcount() // 3
grouping = df.groupby(
    group_columns + ["chunk"],
    as_index=False,
    sort=False,
).agg(
    question_text=("question_text", list),
    answer_text=("answer_text", list),
)
