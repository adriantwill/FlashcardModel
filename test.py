import json

import pandas as pd

df = pd.read_csv("data/sql/questions_rows.csv")
test = df[df["options"].isna()]
print(test)
grouping = df.groupby(as_index=False, by=["storage_path", "page_number"]).agg(
    question_text=("question_text", list),
    answer_text=("answer_text", list),
    options=("options", list),
)
dataset = grouping
row = dataset.iloc[0]
label = []
for i in range(len(row["question_text"])):
    test = json.loads(row["options"][i])
    label.append(
        {
            "question": row["question_text"][i],
            "answer": row["answer_text"][i],
            "options": test,
        }
    )
json_str = json.dumps(label)
print(json_str)
