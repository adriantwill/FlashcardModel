import pandas as pd

def load_data(paths: list[str]):
    df = pd.read_csv(paths)
    for path in paths[1:]:
        df = pd.concat([df, pd.read_csv(path)])
    return df

def filter_data(df, language='en'):
    # Keep only desired rows
    
def clean_data(df):
    # Remove nulls, dupes, invalid
    
def format_for_model(row):
    # Row → training format dict
    
def split_data(df, train=0.8, val=0.1, test=0.1):
    # Return train_df, val_df, test_df
    
def save_to_jsonl(df, path):
    # Write JSONL file

if __name__ == "__main__":
    load_data(['data/raw/ai_flashcards_notes_dataset_v1_part1.csv'])