"""
Data preparation module for flashcard generation.

This module handles loading, filtering, and formatting the 300K flashcard dataset
for training a FLAN-T5 model.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
import random


def load_all_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load and combine all CSV files from the flashcard dataset.

    Args:
        data_dir: Directory containing the CSV files

    Returns:
        Combined DataFrame with all flashcard data
    """
    data_path = Path(data_dir)

    # Load all three parts
    part1 = pd.read_csv(data_path / "ai_flashcards_notes_dataset_v1_part1.csv")
    part2 = pd.read_csv(data_path / "ai_flashcards_notes_dataset_v1_part2.csv")
    part3 = pd.read_csv(data_path / "ai_flashcards_notes_dataset_v1_part3.csv")

    # Combine all parts
    df = pd.concat([part1, part2, part3], ignore_index=True)

    print(f"Loaded {len(df)} total flashcards")
    print(f"Columns: {list(df.columns)}")

    return df


def filter_english(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to only English examples.

    Args:
        df: Combined flashcard DataFrame

    Returns:
        DataFrame with only English examples
    """
    # Filter to English language only
    df_english = df[df['language'] == 'en'].copy()

    print(f"Filtered to {len(df_english)} English examples")

    return df_english


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing invalid entries.

    Args:
        df: Flashcard DataFrame

    Returns:
        Cleaned DataFrame
    """
    initial_count = len(df)

    # Remove rows with missing critical fields
    df = df.dropna(subset=['source_text', 'question', 'answer'])

    # Remove very short source texts (< 50 characters)
    df = df[df['source_text'].str.len() >= 50]

    # Remove duplicates based on source_text
    df = df.drop_duplicates(subset=['source_text'], keep='first')

    removed = initial_count - len(df)
    print(f"Cleaned data: removed {removed} invalid entries, {len(df)} remaining")

    return df


def format_for_training(row: pd.Series) -> Dict[str, str]:
    """
    Transform a single row into the training format.

    Input-target format:
    - Input: source_text
    - Target: "Q: {question} A: {answer}"

    Args:
        row: DataFrame row containing flashcard data

    Returns:
        Dictionary with 'input' and 'target' keys
    """
    input_text = row['source_text'].strip()
    target_text = f"Q: {row['question'].strip()} A: {row['answer'].strip()}"

    return {
        'input': input_text,
        'target': target_text
    }


def create_train_val_test_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.

    Args:
        df: Cleaned flashcard DataFrame
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Split sizes must sum to 1.0"

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['subject'] if 'subject' in df.columns else None
    )

    # Second split: separate train and validation
    adjusted_val_size = val_size / (train_size + val_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=train_val_df['subject'] if 'subject' in train_val_df.columns else None
    )

    print(f"\nDataset split:")
    print(f"  Training:   {len(train_df):,} examples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df):,} examples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:       {len(test_df):,} examples ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def save_to_jsonl(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame to JSONL format for training.

    Each line is a JSON object with 'input' and 'target' fields.

    Args:
        df: DataFrame to save
        output_path: Path to output JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            training_example = format_for_training(row)
            f.write(json.dumps(training_example, ensure_ascii=False) + '\n')

    print(f"Saved {len(df)} examples to {output_path}")


def prepare_dataset(
    data_dir: str = "data/raw",
    output_dir: str = "data/processed",
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42
):
    """
    Complete data preparation pipeline.

    Loads, filters, cleans, splits, and saves the flashcard dataset.

    Args:
        data_dir: Directory containing raw CSV files
        output_dir: Directory to save processed JSONL files
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
    """
    print("=" * 60)
    print("FLASHCARD DATA PREPARATION PIPELINE")
    print("=" * 60)

    # Step 1: Load all data
    print("\n[1/5] Loading data...")
    df = load_all_data(data_dir)

    # Step 2: Filter to English
    print("\n[2/5] Filtering to English examples...")
    df = filter_english(df)

    # Step 3: Clean data
    print("\n[3/5] Cleaning data...")
    df = clean_data(df)

    # Step 4: Create splits
    print("\n[4/5] Creating train/val/test splits...")
    train_df, val_df, test_df = create_train_val_test_split(
        df, train_size, val_size, test_size, random_state
    )

    # Step 5: Save to JSONL
    print("\n[5/5] Saving to JSONL format...")
    save_to_jsonl(train_df, f"{output_dir}/train.jsonl")
    save_to_jsonl(val_df, f"{output_dir}/val.jsonl")
    save_to_jsonl(test_df, f"{output_dir}/test.jsonl")

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 60)

    # Print sample
    print("\nSample training example:")
    sample_row = train_df.iloc[0]
    sample = format_for_training(sample_row)
    print(f"\nINPUT:\n{sample['input']}\n")
    print(f"TARGET:\n{sample['target']}\n")

    return train_df, val_df, test_df


if __name__ == "__main__":
    # Run the complete pipeline
    prepare_dataset()
