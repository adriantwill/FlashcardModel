import json

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r") as file:
        data = json.load(file)

    for item in data["data"]:
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                rows.append(
                    {
                        "context": context,
                        "question": qa["question"],
                        "answer": qa["answers"][0]["text"],
                    }
                )
    return pd.DataFrame(rows)


# def filter_data(df, language='en'):
# Keep only desired rows

# def clean_data(df):
# Remove nulls, dupes, invalid

# def format_for_model(row):
# Row → training format dict


def split_data(
    train_path: str, dev_path: str, split_ratio: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = load_data(train_path)
    dev_df = load_data(dev_path)
    dev_df = dev_df.sample(frac=1).reset_index(drop=True)
    split_idx = int(len(dev_df) * split_ratio)
    val_df = dev_df.iloc[:split_idx]
    test_df = dev_df.iloc[split_idx:]
    return train_df, val_df, test_df


# Return train_df, val_df, test_df

# def save_to_jsonl(df, path):
# Write JSONL file

if __name__ == "__main__":
    