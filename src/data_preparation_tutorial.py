"""
Data Preparation - TUTORIAL VERSION

This is an educational version of data_preparation.py with extensive comments
explaining WHY each step is necessary and HOW it works.

GOAL: Transform 300K flashcard examples from CSV format into training-ready data.

LEARNING OBJECTIVES:
1. Understand data loading and pandas operations
2. Learn data cleaning best practices
3. Understand train/val/test splitting
4. Practice data transformation for ML
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
import random


# ==============================================================================
# SECTION 1: DATA LOADING
# ==============================================================================

def load_all_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load and combine all CSV files from the flashcard dataset.

    WHY: Our data is split across 3 files (100K each). We need to combine them
    into one DataFrame for easier processing.

    HOW:
    - pd.read_csv(): Loads CSV into DataFrame (table structure)
    - pd.concat(): Stacks DataFrames vertically (like stacking papers)
    - ignore_index=True: Renumber rows from 0 after combining

    CONCEPTS:
    - DataFrame: Think of it as an Excel spreadsheet in Python
    - Each row = one flashcard example
    - Each column = one feature (source_text, question, answer, etc.)

    Args:
        data_dir: Directory containing the CSV files

    Returns:
        Combined DataFrame with all flashcard data (should be ~300K rows)

    EXERCISE:
    - Print df.shape after loading - what does (300000, 13) mean?
    - Print df.dtypes - what data type is each column?
    """
    data_path = Path(data_dir)

    # STEP 1: Load each CSV file
    # pd.read_csv() reads CSV and creates a DataFrame
    part1 = pd.read_csv(data_path / "ai_flashcards_notes_dataset_v1_part1.csv")
    part2 = pd.read_csv(data_path / "ai_flashcards_notes_dataset_v1_part2.csv")
    part3 = pd.read_csv(data_path / "ai_flashcards_notes_dataset_v1_part3.csv")

    # STEP 2: Combine into one DataFrame
    # pd.concat([list of DataFrames]) stacks them vertically
    # ignore_index=True renumbers rows: 0, 1, 2, ... instead of 0-99999, 0-99999, 0-99999
    df = pd.concat([part1, part2, part3], ignore_index=True)

    # STEP 3: Verify and report
    print(f"Loaded {len(df)} total flashcards")  # len(df) = number of rows
    print(f"Columns: {list(df.columns)}")  # df.columns = list of column names

    # DEBUGGING TIP: If you get errors here, check:
    # 1. Are all 3 CSV files in data_dir?
    # 2. Do they all have the same columns?
    # 3. Try loading just part1 first to isolate the issue

    return df


# ==============================================================================
# SECTION 2: DATA FILTERING
# ==============================================================================

def filter_english(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to only English examples.

    WHY: Starting with one language simplifies training and evaluation.
    We can add multilingual support later once baseline works.

    HOW:
    - df[condition] is "boolean indexing" - selects rows where condition is True
    - df['language'] == 'en' creates a True/False column
    - df[df['language'] == 'en'] keeps only True rows

    CONCEPT: Boolean Indexing
    Think of it as a filter:
    - For each row, check: Is language column = 'en'?
    - If yes, keep row
    - If no, discard row

    Args:
        df: Combined flashcard DataFrame

    Returns:
        DataFrame with only English examples (~100K rows)

    EXERCISE:
    - How would you filter for 'hard' difficulty only?
    - How would you filter for Math OR Physics?
    - Print value_counts() of different columns to explore
    """
    # STEP 1: Create boolean mask (True/False for each row)
    # mask = df['language'] == 'en'
    # This creates a series like: [True, True, False, True, ...]

    # STEP 2: Apply mask to filter rows
    # df[mask] keeps only rows where mask is True
    df_english = df[df['language'] == 'en'].copy()

    # .copy() creates a new DataFrame (avoids warnings about modifying views)

    print(f"Filtered to {len(df_english)} English examples")
    print(f"That's {len(df_english)/len(df)*100:.1f}% of the original data")

    # DEBUGGING TIP: If you get 0 rows:
    # 1. Check df['language'].unique() - what values actually exist?
    # 2. Maybe it's 'EN' or 'english' instead of 'en'?

    return df_english


# ==============================================================================
# SECTION 3: DATA CLEANING
# ==============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing invalid entries.

    WHY: "Garbage in, garbage out"
    - Missing data confuses the model
    - Very short texts don't have enough information
    - Duplicates waste training time and cause overfitting

    HOW:
    - dropna(): Remove rows with missing values in critical columns
    - Boolean indexing: Filter based on text length
    - drop_duplicates(): Remove duplicate rows

    CONCEPT: Data Quality Matters
    A model trained on clean data with 80K examples will outperform
    a model trained on messy data with 100K examples!

    Args:
        df: Flashcard DataFrame

    Returns:
        Cleaned DataFrame

    EXERCISE:
    - What if we want to keep short texts? How would you modify this?
    - How would you check for empty strings ('' instead of NaN)?
    - Plot distribution of source_text lengths before/after cleaning
    """
    initial_count = len(df)

    # STEP 1: Remove rows with missing critical fields
    # dropna(subset=['col1', 'col2']) removes rows where col1 OR col2 is NaN
    # We need source_text, question, AND answer for training
    df = df.dropna(subset=['source_text', 'question', 'answer'])

    missing_removed = initial_count - len(df)
    print(f"Removed {missing_removed} rows with missing data")

    # STEP 2: Remove very short source texts
    # Why 50 characters? If text is too short, can't make a good flashcard
    # .str.len() gets length of each string in the column
    df = df[df['source_text'].str.len() >= 50]

    short_removed = (initial_count - missing_removed) - len(df)
    print(f"Removed {short_removed} rows with very short source texts")

    # STEP 3: Remove duplicates based on source_text
    # Why? If same paragraph appears twice, model just memorizes it
    # keep='first' keeps the first occurrence, removes others
    df = df.drop_duplicates(subset=['source_text'], keep='first')

    duplicates_removed = (initial_count - missing_removed - short_removed) - len(df)
    print(f"Removed {duplicates_removed} duplicate source texts")

    # STEP 4: Report final state
    total_removed = initial_count - len(df)
    print(f"\nCleaning summary:")
    print(f"  Started with: {initial_count:,} rows")
    print(f"  Removed: {total_removed:,} rows ({total_removed/initial_count*100:.1f}%)")
    print(f"  Final: {len(df):,} clean rows")

    # DEBUGGING TIP: If you lose too many rows:
    # 1. Check how many have missing data: df.isnull().sum()
    # 2. Check text length distribution: df['source_text'].str.len().describe()
    # 3. Adjust thresholds based on your data

    return df


# ==============================================================================
# SECTION 4: DATA TRANSFORMATION
# ==============================================================================

def format_for_training(row: pd.Series) -> Dict[str, str]:
    """
    Transform a single row into the training format.

    WHY: Models need data in a specific format. We're doing seq2seq:
    - Input: source text (what the model reads)
    - Target: Q&A combined (what the model should generate)

    HOW: Extract relevant columns and format as needed

    CONCEPT: Task Design
    This is YOUR choice! Different formats teach different behaviors:
    - Format 1: "Q: ... A: ..." (what we use - simple and clear)
    - Format 2: "Question: ... Answer: ..." (more verbose)
    - Format 3: "<question> ... </question> <answer> ... </answer>" (XML-like)

    The model will learn whatever pattern you show it!

    Args:
        row: DataFrame row (pd.Series) containing flashcard data

    Returns:
        Dictionary with 'input' and 'target' keys

    EXERCISE:
    - Modify to include difficulty: "[HARD] Q: ... A: ..."
    - Try different separators: "QUESTION: ... ANSWER: ..."
    - Add instruction prefix: "Generate flashcard: [text]"
    - Compare which format gives best results!
    """
    # STEP 1: Extract source text (INPUT)
    # .strip() removes leading/trailing whitespace
    input_text = row['source_text'].strip()

    # STEP 2: Combine question and answer (TARGET)
    # This is the pattern the model will learn to generate
    question = row['question'].strip()
    answer = row['answer'].strip()
    target_text = f"Q: {question} A: {answer}"

    # ALTERNATIVE FORMATS (uncomment to try):
    # target_text = f"Question: {question} Answer: {answer}"
    # target_text = f"[{row['difficulty'].upper()}] Q: {question} A: {answer}"
    # target_text = json.dumps({'question': question, 'answer': answer})

    # STEP 3: Return as dictionary
    return {
        'input': input_text,
        'target': target_text
    }

    # CONCEPT: Why dictionary?
    # - HuggingFace datasets library expects this format
    # - Easy to convert to JSON
    # - Clear key names (input vs target)


# ==============================================================================
# SECTION 5: TRAIN/VAL/TEST SPLITTING
# ==============================================================================

def create_train_val_test_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.

    WHY: Need separate data for different purposes:
    - Train: Model learns from this (updates weights)
    - Validation: Check progress during training (tune hyperparameters)
    - Test: Final evaluation (simulate real-world performance)

    HOW: Use sklearn's train_test_split twice
    - First split: train+val vs test
    - Second split: train vs val

    CONCEPT: Why 3 splits instead of 2?

    Imagine you're studying for an exam:
    - Training set = textbook (you study from this)
    - Validation set = practice problems (check understanding, adjust study method)
    - Test set = actual exam (final grade, never seen before)

    If you only had train + test, you'd keep trying different study methods
    and testing on the exam. Eventually you'd memorize the exam questions!
    Validation set lets you tune your approach without cheating on the test.

    Args:
        df: Cleaned flashcard DataFrame
        train_size: Proportion for training (0.8 = 80%)
        val_size: Proportion for validation (0.1 = 10%)
        test_size: Proportion for test (0.1 = 10%)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)

    EXERCISE:
    - Try 90/5/5 split - does it improve results?
    - Try 50/25/25 - does it hurt?
    - Plot learning curves with different splits
    - What happens without stratification?
    """
    # STEP 0: Verify sizes sum to 1.0
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Split sizes must sum to 1.0"

    # STEP 1: First split - separate test set
    # stratify=df['subject'] ensures each split has similar subject distribution
    # Example: If 30% of data is Math, test set will also be ~30% Math
    # This prevents bias (e.g., test set being all Physics)
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,        # Hold out 10% for test
        random_state=random_state,   # Seed for reproducibility
        stratify=df['subject'] if 'subject' in df.columns else None  # Balance subjects
    )

    # STEP 2: Second split - separate train and validation
    # Now we split the remaining 90% into 80% train and 10% val
    # adjusted_val_size = 0.1 / 0.9 = 0.111... (10% of remaining 90%)
    adjusted_val_size = val_size / (train_size + val_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=train_val_df['subject'] if 'subject' in train_val_df.columns else None
    )

    # STEP 3: Report split sizes
    print(f"\nDataset split:")
    print(f"  Training:   {len(train_df):,} examples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df):,} examples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:       {len(test_df):,} examples ({len(test_df)/len(df)*100:.1f}%)")

    # CONCEPT: Random State / Seed
    # random_state=42 makes the split reproducible
    # Same seed = same split every time
    # Different seed = different split
    # Why? So you can share results with others and they get the same split!

    # DEBUGGING TIP: Verify splits don't overlap
    # train_ids = set(train_df['id'])
    # val_ids = set(val_df['id'])
    # test_ids = set(test_df['id'])
    # assert len(train_ids & val_ids) == 0, "Train and val overlap!"
    # assert len(train_ids & test_ids) == 0, "Train and test overlap!"
    # assert len(val_ids & test_ids) == 0, "Val and test overlap!"

    return train_df, val_df, test_df


# ==============================================================================
# SECTION 6: SAVING TO JSONL
# ==============================================================================

def save_to_jsonl(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame to JSONL format for training.

    WHY JSONL (not JSON or CSV)?
    - JSON: Loads entire file into memory (100K examples = huge!)
    - CSV: Hard to handle nested structures, special characters
    - JSONL: Each line is a JSON object
      - Stream line-by-line (memory efficient)
      - Easy to parse
      - Standard for ML datasets

    JSONL Example:
    {"input": "Text 1...", "target": "Q: ... A: ..."}
    {"input": "Text 2...", "target": "Q: ... A: ..."}
    {"input": "Text 3...", "target": "Q: ... A: ..."}

    Each line is independent - can read one at a time!

    Args:
        df: DataFrame to save
        output_path: Path to output JSONL file

    EXERCISE:
    - Load the JSONL file line by line and print first 3
    - Compare file size of JSONL vs CSV vs JSON
    - Write a function to merge multiple JSONL files
    """
    output_path = Path(output_path)

    # Create parent directories if they don't exist
    # exist_ok=True prevents error if directory already exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # STEP 1: Open file for writing
    # 'w' = write mode (overwrites existing file)
    # encoding='utf-8' handles special characters (é, ñ, 中, etc.)
    with open(output_path, 'w', encoding='utf-8') as f:

        # STEP 2: Iterate through each row
        # iterrows() returns (index, row) for each row
        # We ignore index with _
        for _, row in df.iterrows():

            # STEP 3: Transform row to training format
            training_example = format_for_training(row)

            # STEP 4: Write as JSON line
            # json.dumps() converts Python dict to JSON string
            # ensure_ascii=False preserves non-English characters
            # '\n' adds newline after each JSON object
            f.write(json.dumps(training_example, ensure_ascii=False) + '\n')

    print(f"Saved {len(df)} examples to {output_path}")

    # CONCEPT: Why iterate instead of df.to_json()?
    # - We need to transform each row with format_for_training()
    # - to_json() doesn't allow custom transformation
    # - Manual iteration gives us full control

    # DEBUGGING TIP: Verify file format
    # with open(output_path) as f:
    #     line = f.readline()
    #     obj = json.loads(line)
    #     assert 'input' in obj and 'target' in obj, "Missing keys!"


# ==============================================================================
# SECTION 7: COMPLETE PIPELINE
# ==============================================================================

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

    This orchestrates all the steps:
    1. Load → 2. Filter → 3. Clean → 4. Split → 5. Save

    WHY: Organize everything in one place
    - Easy to run entire pipeline
    - Clear, linear flow
    - Can swap out steps easily

    CONCEPT: Pipeline Design
    Each function does ONE thing well:
    - load_all_data: just loads
    - filter_english: just filters
    - clean_data: just cleans
    - etc.

    This is modular design - easy to test, debug, and modify!

    Args:
        data_dir: Directory containing raw CSV files
        output_dir: Directory to save processed JSONL files
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_state: Random seed for reproducibility

    EXERCISE:
    - Add a step to remove rare subjects (< 1000 examples)
    - Add data augmentation (paraphrase source texts)
    - Add quality filtering (remove low-quality Q&A)
    - Save statistics to a JSON file
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

    # CONCEPT: Why print sample?
    # Always inspect your data!
    # - Does the format look right?
    # - Is the input text clean?
    # - Does the target follow Q:/A: format?
    # Catching errors here saves hours of debugging later!

    return train_df, val_df, test_df


# ==============================================================================
# MAIN: RUN THE PIPELINE
# ==============================================================================

if __name__ == "__main__":
    """
    This runs when you execute: python data_preparation_tutorial.py

    EXERCISE: Before running, try to predict:
    1. How many English examples will there be? (~100K? ~200K?)
    2. How many will be removed in cleaning? (1%? 10%?)
    3. What will be the exact train/val/test sizes?

    Then run and check your predictions!
    """
    # Run the complete pipeline
    prepare_dataset()

    # NEXT STEPS:
    # 1. Open data/processed/train.jsonl and inspect manually
    # 2. Load train.jsonl in Python and print 10 random examples
    # 3. Check: Are there any patterns in the data you didn't expect?
    # 4. Move on to model training!
