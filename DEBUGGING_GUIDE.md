# Debugging Guide - Common Mistakes & Solutions

**Goal:** Help you debug issues quickly and learn from mistakes.

**Philosophy:** Errors are learning opportunities! Each bug you fix teaches you something new.

---

## General Debugging Strategy

### 1. Read the Error Message

**What to look for:**
- **Error type:** `ValueError`, `KeyError`, `RuntimeError`, etc.
- **Line number:** Where did it fail?
- **Message:** What went wrong?

**Example:**
```
KeyError: 'language'
  File "data_preparation.py", line 45, in filter_english
    df_english = df[df['language'] == 'en']
```

**Diagnosis:**
- Error type: `KeyError` - trying to access a column that doesn't exist
- Line 45: `df['language']`
- Problem: DataFrame doesn't have a 'language' column

**Solution:**
```python
# Check what columns actually exist
print(df.columns)

# Maybe the column is named differently?
print([col for col in df.columns if 'lang' in col.lower()])
```

### 2. Print Debugging

**Add strategic print statements:**

```python
# Before the error
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"First row: {df.iloc[0]}")

# Your code that fails
result = some_function(df)
```

### 3. Isolate the Problem

**Test in isolation:**

```python
# Instead of running the whole pipeline
# Test just one function

df = load_all_data()  # This works
print(f"Loaded: {len(df)} rows")  # ✓

df_en = filter_english(df)  # This fails
# Now you know the problem is in filter_english
```

### 4. Check Your Assumptions

**Common assumptions that fail:**
- "The file exists" → Check with `Path(file).exists()`
- "The column exists" → Check with `'col' in df.columns`
- "The data is clean" → Check with `df.info()`, `df.describe()`
- "The shape is right" → Check with `tensor.shape`

---

## Common Errors by Category

### Data Loading Errors

#### Error: `FileNotFoundError: [Errno 2] No such file or directory`

**Cause:** File doesn't exist at the specified path

**Debug:**
```python
from pathlib import Path

file_path = Path("data/raw/dataset.csv")
print(f"File exists: {file_path.exists()}")
print(f"Parent dir exists: {file_path.parent.exists()}")
print(f"Files in parent dir: {list(file_path.parent.glob('*'))}")
```

**Solutions:**
1. Check the path is correct
2. Check you're in the right directory (`os.getcwd()`)
3. Check file extension (.csv vs .CSV)
4. Use absolute paths for clarity

#### Error: `UnicodeDecodeError: 'utf-8' codec can't decode`

**Cause:** File has non-UTF-8 encoding

**Solution:**
```python
# Try different encodings
df = pd.read_csv(file_path, encoding='latin-1')
# or
df = pd.read_csv(file_path, encoding='cp1252')
```

#### Error: `ParserError: Error tokenizing data`

**Cause:** Malformed CSV (inconsistent columns, quotes, etc.)

**Debug:**
```python
# Read line by line to find problematic row
with open(file_path) as f:
    for i, line in enumerate(f):
        try:
            # Process line
            pass
        except Exception as e:
            print(f"Error at line {i}: {e}")
            print(f"Line content: {line}")
            break
```

**Solution:**
```python
# Be more lenient
df = pd.read_csv(file_path, on_bad_lines='skip', encoding_errors='ignore')
```

---

### DataFrame Errors

#### Error: `KeyError: 'column_name'`

**Cause:** Column doesn't exist

**Debug:**
```python
print(f"Available columns: {df.columns.tolist()}")
print(f"Looking for: 'column_name'")

# Case-sensitive check
print(f"Exact match: {'column_name' in df.columns}")

# Fuzzy search
matches = [col for col in df.columns if 'column' in col.lower()]
print(f"Similar columns: {matches}")
```

**Solution:**
```python
# Use the correct column name
# or rename:
df = df.rename(columns={'Column_Name': 'column_name'})
```

#### Error: `ValueError: Length mismatch`

**Cause:** Trying to assign a Series/list of wrong length

**Debug:**
```python
print(f"DataFrame length: {len(df)}")
print(f"New column length: {len(new_values)}")
```

**Solution:**
```python
# Make sure lengths match
assert len(new_values) == len(df), "Length mismatch!"
df['new_col'] = new_values
```

#### Error: `SettingWithCopyWarning`

**Cause:** Modifying a DataFrame slice (might not work as expected)

**Example:**
```python
df_subset = df[df['language'] == 'en']  # This is a view, not a copy
df_subset['new_col'] = values  # ⚠️ Warning!
```

**Solution:**
```python
# Explicitly copy
df_subset = df[df['language'] == 'en'].copy()
df_subset['new_col'] = values  # ✓ No warning
```

---

### Training Errors

#### Error: `CUDA out of memory`

**Cause:** Batch size too large for GPU memory

**Debug:**
```python
# Check GPU memory
import torch
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

**Solutions:**
1. **Reduce batch size:**
   ```python
   per_device_train_batch_size = 4  # instead of 8
   ```

2. **Increase gradient accumulation:**
   ```python
   gradient_accumulation_steps = 8  # instead of 4
   # Effective batch size stays the same: 4*8 = 32
   ```

3. **Reduce sequence length:**
   ```python
   max_input_length = 256  # instead of 512
   ```

4. **Use gradient checkpointing:**
   ```python
   model.gradient_checkpointing_enable()
   # Trades compute for memory
   ```

5. **Use CPU instead:**
   ```python
   device = 'cpu'
   # Much slower but won't OOM
   ```

#### Error: `RuntimeError: Expected tensor of size [8, 512], got [8, 256]`

**Cause:** Tensor shape mismatch

**Debug:**
```python
# Print shapes at each step
print(f"Input shape: {inputs.shape}")
print(f"Expected: [batch_size, max_length]")

# Check tokenizer settings
print(f"Tokenizer max_length: {tokenizer.model_max_length}")
```

**Solution:**
```python
# Ensure consistent max_length
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(text, max_length=512, truncation=True, padding='max_length')
```

#### Error: `Loss is NaN`

**Cause:** Gradient explosion, learning rate too high, or bad data

**Debug:**
```python
# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")

# Check data
for batch in dataloader:
    print(f"Input IDs range: {batch['input_ids'].min()} to {batch['input_ids'].max()}")
    print(f"Labels range: {batch['labels'].min()} to {batch['labels'].max()}")
    break
```

**Solutions:**
1. **Lower learning rate:**
   ```python
   learning_rate = 1e-5  # instead of 5e-5
   ```

2. **Enable gradient clipping:**
   ```python
   max_grad_norm = 1.0  # in TrainingArguments
   ```

3. **Check for bad data:**
   ```python
   # Remove NaN/Inf values
   df = df.dropna()
   ```

4. **Use mixed precision carefully:**
   ```python
   fp16 = False  # Disable if causing issues
   ```

#### Error: `Training loss not decreasing`

**Cause:** Learning rate too low, model bug, or data issue

**Debug:**
```python
# Overfit on small sample (sanity check)
small_dataset = train_dataset.select(range(100))
trainer = Trainer(train_dataset=small_dataset, ...)
trainer.train()

# Loss should go to near-zero on 100 examples
# If not, there's a fundamental problem
```

**Solutions:**
1. **Increase learning rate:**
   ```python
   learning_rate = 1e-4  # instead of 1e-5
   ```

2. **Train longer:**
   ```python
   num_train_epochs = 5  # instead of 3
   ```

3. **Check data format:**
   ```python
   # Print a few examples
   for i in range(3):
       example = train_dataset[i]
       print(f"Input: {example['input']}")
       print(f"Target: {example['target']}")
       # Do they look correct?
   ```

4. **Verify model is training:**
   ```python
   model.train()  # Not model.eval()
   for param in model.parameters():
       assert param.requires_grad, "Parameters not trainable!"
   ```

---

### Generation Errors

#### Error: `Model outputs gibberish`

**Cause:** Model not trained, wrong tokenizer, or bad generation params

**Debug:**
```python
# Check model was trained
print(f"Model weights changed: {model.state_dict()['some_layer.weight'][0]}")

# Check tokenizer matches
print(f"Tokenizer: {tokenizer.__class__.__name__}")
print(f"Model expects: {model.config.name_or_path}")

# Try different generation settings
output1 = model.generate(..., temperature=0.5, num_beams=1)  # Greedy
output2 = model.generate(..., temperature=0.7, num_beams=4)  # Beam search
output3 = model.generate(..., temperature=1.0, num_beams=4, do_sample=True)  # Sampling
```

**Solutions:**
1. **Verify model trained:**
   ```python
   # Check training loss decreased
   print(trainer.state.log_history)
   ```

2. **Use correct tokenizer:**
   ```python
   # Must match the model
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   ```

3. **Adjust generation params:**
   ```python
   output = model.generate(
       input_ids,
       max_length=256,
       num_beams=4,
       temperature=0.7,
       no_repeat_ngram_size=3,
       early_stopping=True
   )
   ```

#### Error: `Model always generates the same output`

**Cause:** Sampling disabled, temperature too low, or model collapsed

**Debug:**
```python
# Generate multiple times
for i in range(5):
    output = model.generate(input_ids, ...)
    print(f"Output {i}: {tokenizer.decode(output[0])}")
# Are they all identical?
```

**Solutions:**
1. **Enable sampling:**
   ```python
   output = model.generate(
       input_ids,
       do_sample=True,
       top_p=0.9,
       temperature=1.0
   )
   ```

2. **Increase temperature:**
   ```python
   temperature = 0.9  # instead of 0.5
   ```

3. **Check model hasn't collapsed:**
   ```python
   # Validation loss should be reasonable
   eval_results = trainer.evaluate()
   print(f"Val loss: {eval_results['eval_loss']}")
   # If > 10, model likely collapsed
   ```

---

### PDF Processing Errors

#### Error: `Text extraction is scrambled`

**Cause:** Multi-column layout, complex formatting

**Debug:**
```python
# Extract and print raw text
text = extract_text_from_pdf(pdf_path)
print(text[:500])  # First 500 characters
# Is it readable?
```

**Solutions:**
1. **Try different library:**
   ```python
   # Use PyPDF2 instead of pdfplumber
   from PyPDF2 import PdfReader
   reader = PdfReader(pdf_path)
   text = ''.join(page.extract_text() for page in reader.pages)
   ```

2. **Use OCR for scanned PDFs:**
   ```python
   # pytesseract for OCR
   # pdf2image to convert PDF to images
   ```

3. **Preprocess PDF:**
   - Convert multi-column to single column
   - Remove headers/footers manually
   - Split by sections

#### Error: `No text extracted from PDF`

**Cause:** Scanned PDF (images, not text), or encrypted

**Debug:**
```python
with pdfplumber.open(pdf_path) as pdf:
    print(f"Number of pages: {len(pdf.pages)}")
    page = pdf.pages[0]
    print(f"Text found: {bool(page.extract_text())}")
    print(f"Images found: {len(page.images)}")
```

**Solutions:**
1. **Check if scanned:**
   - If `images found > 0` and `text found = False`, it's scanned
   - Use OCR

2. **Check if encrypted:**
   ```python
   from PyPDF2 import PdfReader
   reader = PdfReader(pdf_path)
   if reader.is_encrypted:
       reader.decrypt(password)
   ```

---

## Debugging Checklist

Before asking for help, check:

- [ ] Read the error message carefully
- [ ] Googled the error message
- [ ] Checked file paths exist
- [ ] Verified data format (print first few rows)
- [ ] Checked data types (df.dtypes, tensor.dtype)
- [ ] Checked shapes (df.shape, tensor.shape)
- [ ] Tried on a small sample (first 100 examples)
- [ ] Checked for typos in column/variable names
- [ ] Verified package versions (`pip list`)
- [ ] Restarted kernel/Python interpreter
- [ ] Read the relevant section of LEARNING_GUIDE.md

---

## Tools for Debugging

### Python Debugger (pdb)

```python
import pdb

def my_function(x):
    pdb.set_trace()  # Execution stops here
    result = x * 2
    return result

# When you run this, you'll get an interactive prompt
# Commands:
# - n (next): Execute next line
# - s (step): Step into function
# - c (continue): Continue execution
# - p variable: Print variable
# - q (quit): Quit debugger
```

### IPython/Jupyter

```python
# Automatic debugger on exception
%pdb on

# Time code execution
%timeit my_function(x)

# Profile memory usage
%memit my_function(x)

# Interactive debugging
from IPython.core.debugger import set_trace
set_trace()
```

### Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_data(df):
    logger.debug(f"Input shape: {df.shape}")
    df_clean = clean_data(df)
    logger.debug(f"After cleaning: {df_clean.shape}")
    return df_clean
```

---

## Getting Help

### Where to Ask

1. **Stack Overflow**
   - Tag: `[python] [pandas] [transformers]`
   - Provide minimal reproducible example
   - Show what you've tried

2. **HuggingFace Forums**
   - For model/training issues
   - Very helpful community
   - https://discuss.huggingface.co/

3. **GitHub Issues**
   - For package-specific bugs
   - Check if issue already exists
   - Provide full error trace

### How to Ask

**Good question:**
```
Title: KeyError when filtering DataFrame by language column

I'm trying to filter a DataFrame to English examples:

```python
df_english = df[df['language'] == 'en']
```

But I get: KeyError: 'language'

I've checked:
- df.columns shows: ['id', 'text', 'question', 'answer']
- The file is ai_flashcards_notes_dataset_v1_part1.csv
- pandas version: 2.1.4

Is the column named differently? How can I find it?
```

**Bad question:**
```
Title: Code doesn't work

My code has an error. Please help!

[No code, no error message, no context]
```

---

## Learning from Errors

**Keep a debugging log:**

```markdown
## Error: CUDA OOM

Date: 2025-01-15
Context: Training FLAN-T5-base

What went wrong:
- CUDA out of memory error
- Batch size was 8

What I tried:
1. Reduced batch size to 4 - still OOM
2. Reduced sequence length to 256 - worked!

What I learned:
- Sequence length matters more than batch size for memory
- 512 tokens × 8 batch = 4096 total tokens
- 256 tokens × 8 batch = 2048 total tokens (fits!)

Future: Start with smaller sequence length, then increase
```

**This helps you:**
1. Remember solutions for next time
2. Build intuition about what works
3. Track your learning progress

---

## Summary

**Remember:**
- Errors are normal and expected
- Each error teaches you something
- The solution is often simpler than you think
- Read. Print. Debug. Repeat.

**Most common fixes:**
1. Check the path/file exists
2. Print the shape/type
3. Verify the data format
4. Reduce the scale (test on small sample)
5. Google the error message

**You've got this!** 💪

Happy debugging!
