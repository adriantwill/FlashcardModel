# Common Fixes for Training Issues

## OverflowError during evaluation

### Error Message:
```
OverflowError: out of range integral type conversion attempted
in _decode(self, token_ids, skip_special_tokens, clean_up_tokenization_spaces, **kwargs)
```

### Cause:
The tokenizer is trying to decode `-100` values (used for label masking) which are outside the valid token ID range.

### Fix:

In your `compute_metrics` function, ensure you replace `-100` with the pad token ID BEFORE decoding:

**WRONG (causes error):**
```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # ❌ This will fail - labels still contain -100
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ...
```

**CORRECT:**
```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # ✓ Replace -100 with pad_token_id BEFORE decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Now compute metrics
    rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    # ...
```

### Quick Fix for Notebooks:

If you're in a Colab/Jupyter notebook, add this cell and re-run from the metrics section:

```python
import numpy as np

# Fixed compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # IMPORTANT: Replace -100 in labels before decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE
    rouge_scores = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    # Compute BLEU
    bleu_score = bleu.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
    )

    return {
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
        'bleu': bleu_score['bleu']
    }

# Re-initialize trainer with fixed function
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics  # ← Using fixed function
)

# Now evaluation should work
test_results = trainer.evaluate(tokenized_test)
print(test_results)
```

---

## Other Common Training Errors

### 1. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Fix:**
```python
# Reduce batch size
per_device_train_batch_size = 4  # instead of 8
gradient_accumulation_steps = 8  # instead of 4

# Or reduce sequence length
max_input_length = 256  # instead of 512
```

### 2. Loss is NaN

**Error:** Training loss shows `nan`

**Possible causes:**
- Learning rate too high
- Gradient explosion
- Bad data

**Fix:**
```python
# Lower learning rate
learning_rate = 1e-5  # instead of 5e-5

# Enable gradient clipping (should already be in TrainingArguments)
max_grad_norm = 1.0

# Check for bad data
print(train_dataset[0])  # Should look reasonable
```

### 3. Model Not Learning (Loss Not Decreasing)

**Symptom:** Loss stays constant or fluctuates

**Debug:**
```python
# Overfit test - train on tiny dataset
tiny_dataset = train_dataset.select(range(10))
trainer = Seq2SeqTrainer(train_dataset=tiny_dataset, ...)
trainer.train()

# Loss should go to near-zero on 10 examples
# If not, there's a bug in the setup
```

**Common causes:**
- Learning rate too low (increase to 1e-4)
- Wrong labels format
- Model in eval mode instead of train mode

### 4. KeyError: 'input_ids'

**Error:** `KeyError: 'input_ids'`

**Cause:** Data not properly tokenized

**Fix:**
```python
# Verify tokenization
sample = train_dataset[0]
print(sample.keys())  # Should include 'input_ids', 'attention_mask', 'labels'

# Check tokenizer is working
test_text = "Hello world"
tokens = tokenizer(test_text, return_tensors='pt')
print(tokens.keys())  # Should include 'input_ids', 'attention_mask'
```

### 5. Shape Mismatch

**Error:** `RuntimeError: The size of tensor a (512) must match the size of tensor b (256)`

**Cause:** Inconsistent max_length settings

**Fix:**
```python
# Ensure consistent lengths in preprocessing
def preprocess_function(examples):
    inputs = tokenizer(
        examples['input'],
        max_length=512,  # ← Consistent
        truncation=True,
        padding='max_length'
    )
    targets = tokenizer(
        examples['target'],
        max_length=256,  # ← Consistent
        truncation=True,
        padding='max_length'
    )
    inputs['labels'] = targets['input_ids']
    return inputs
```

---

## Debugging Checklist

When you get an error:

1. **Read the error message**
   - What's the error type?
   - Which line failed?
   - What was it trying to do?

2. **Check the traceback**
   - Start from YOUR code (top of traceback)
   - What function did you call?
   - What arguments did you pass?

3. **Print intermediate values**
   ```python
   print(f"predictions shape: {predictions.shape}")
   print(f"labels shape: {labels.shape}")
   print(f"predictions sample: {predictions[0][:10]}")
   print(f"labels sample: {labels[0][:10]}")
   ```

4. **Check data format**
   ```python
   # Print first example
   print(train_dataset[0])

   # After tokenization
   print(tokenized_train[0])
   ```

5. **Isolate the problem**
   ```python
   # Test each step separately
   # Does tokenization work?
   test = tokenizer("Hello", return_tensors='pt')
   print(test)

   # Does model forward pass work?
   output = model(**test)
   print(output)
   ```

---

## Prevention Tips

1. **Always validate data after each transformation**
   ```python
   # After loading
   print(f"Loaded {len(train_dataset)} examples")
   print(train_dataset[0])

   # After tokenizing
   print(f"Tokenized {len(tokenized_train)} examples")
   print(tokenized_train[0].keys())
   ```

2. **Start small**
   ```python
   # Test on 10 examples first
   tiny = train_dataset.select(range(10))
   # If this works, scale up
   ```

3. **Use try-except for debugging**
   ```python
   try:
       result = trainer.evaluate()
   except Exception as e:
       print(f"Error type: {type(e).__name__}")
       print(f"Error message: {e}")
       import traceback
       traceback.print_exc()
   ```

4. **Check before decoding**
   ```python
   # Before calling tokenizer.decode()
   print(f"Min token ID: {token_ids.min()}")
   print(f"Max token ID: {token_ids.max()}")
   print(f"Vocab size: {tokenizer.vocab_size}")

   # token_ids should be in range [0, vocab_size-1]
   # If you see -100, replace it first!
   ```

---

## Still Stuck?

1. **Check DEBUGGING_GUIDE.md** for detailed troubleshooting
2. **Google the exact error message**
3. **Ask on HuggingFace forums** (include error + minimal code)
4. **Check model card** on HuggingFace hub for known issues

---

**Remember:** Every error teaches you something! Don't get discouraged. 💪
