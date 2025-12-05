# Google Colab Survival Guide

## Handling Disconnections

### Problem: Colab Disconnects During Training

**Free tier limits:**
- Maximum session: 12 hours
- Idle timeout: 90 minutes
- No guarantee of GPU availability

### Solution 1: Checkpointing (Automatic)

The training notebook already saves checkpoints! Here's what happens:

```python
# In TrainingArguments
save_strategy = 'steps'
save_steps = 1000  # Save every 1000 steps
save_total_limit = 3  # Keep only 3 most recent checkpoints
```

**What this means:**
- Every 1000 training steps, model is saved to Google Drive
- If disconnect happens, you can resume from last checkpoint
- Not starting from scratch!

### How to Resume After Disconnect:

1. **Check what was saved:**
   ```python
   # In Colab, list checkpoints
   !ls /content/drive/MyDrive/FlashcardMaker/models/flan-t5-flashcard-v1/

   # You'll see:
   # checkpoint-1000/
   # checkpoint-2000/
   # checkpoint-3000/
   ```

2. **Resume training:**
   ```python
   # Instead of starting fresh, resume from checkpoint
   trainer = Seq2SeqTrainer(...)

   # Resume from last checkpoint
   trainer.train(resume_from_checkpoint=True)

   # Or specify a specific checkpoint
   trainer.train(resume_from_checkpoint='checkpoint-3000')
   ```

### Solution 2: Prevent Idle Disconnects

**JavaScript trick** (run in Colab):

```javascript
// Paste this in browser console (F12)
function ClickConnect(){
    console.log("Clicking connect button");
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60000)  // Click every 60 seconds
```

**Or use this Python cell:**

```python
# Add this cell at the end of your notebook
# Keeps session alive by printing every 5 minutes
import time
from IPython.display import clear_output

for i in range(144):  # 144 * 5 min = 12 hours
    time.sleep(300)  # 5 minutes
    clear_output(wait=True)
    print(f"Keeping alive... {i*5} minutes elapsed")
```

### Solution 3: Use Colab Pro (If Affordable)

**Colab Pro ($10/month):**
- Longer sessions (24 hours)
- Better GPU (A100, V100)
- Background execution (can close browser!)
- Priority access

**Worth it if:**
- You're training frequently
- You value your time
- You want better GPUs

---

## 2. Training on M1 Pro (16GB RAM)

**Short answer:** YES, absolutely possible! M1 is actually great for this.

### Why M1 Pro is Good for This:

✅ **16GB unified memory** - Shared between CPU and GPU
✅ **Metal acceleration** - Apple's GPU framework
✅ **Power efficient** - Won't overheat like Intel Macs
✅ **Fast enough** - Training will work, just slower than cloud GPU

### Expected Performance:

| Setup | Training Time (3 epochs, 80K examples) |
|-------|---------------------------------------|
| Colab T4 GPU | 3-4 hours |
| **M1 Pro (your Mac)** | **8-12 hours** |
| CPU only | 24-48 hours |

**Verdict:** Overnight training is totally feasible!

### How to Train on M1 Pro:

#### Step 1: Install MPS Backend for PyTorch

```bash
# Install PyTorch with Metal Performance Shaders (MPS)
pip install --upgrade torch torchvision torchaudio

# Verify MPS is available
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
# Should print: MPS available: True
```

#### Step 2: Modify Training Code

I'll create an M1-optimized version:

```python
# In your training script or notebook

import torch

# Check what's available
if torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon GPU
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
    print("Using CUDA GPU")
else:
    device = "cpu"
    print("Using CPU")

# Important: Disable fp16 for MPS (not fully supported yet)
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/flan-t5-flashcard-v1",
    per_device_train_batch_size=4,  # Smaller for M1
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,  # Compensate for smaller batch
    learning_rate=5e-5,
    num_train_epochs=3,

    # MPS-specific settings
    fp16=False,  # Don't use fp16 on MPS
    # MPS doesn't need this, it auto-manages memory

    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",

    logging_steps=100,
    report_to="none",
    predict_with_generate=True,
)

# Model will automatically use MPS device
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
# No need to explicitly move to device, trainer handles it
```

#### Step 3: Optimize for M1

**Memory management:**

```python
# If you get memory errors, reduce batch size
per_device_train_batch_size = 2  # Instead of 4
gradient_accumulation_steps = 16  # Instead of 8
# Effective batch size stays same: 2 * 16 = 32

# Or reduce sequence length
max_input_length = 384  # Instead of 512
max_target_length = 192  # Instead of 256
```

**Monitor memory usage:**

```python
# Add to training script
import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {mem_mb:.0f} MB / 16384 MB")

# Call periodically during training
```

### Advantages of Local Training:

✅ **Unlimited time** - No 12-hour limit
✅ **No internet needed** - Train offline
✅ **Keep your data private** - Doesn't leave your machine
✅ **Instant iteration** - No upload/download to Colab
✅ **Learn better** - See real-time progress
✅ **Free** - No Colab Pro costs

### Disadvantages:

❌ **Slower** - 2-3x slower than T4 GPU
❌ **Ties up your Mac** - Can't do heavy tasks while training
❌ **Battery drain** - Keep plugged in
❌ **Heat** - Mac will get warm (but M1 handles it well)

---

## Recommended Workflow

### For Learning & Iteration:

**Use your M1 Pro locally:**
```bash
# Start small for experiments
# Train on 10K examples (10-20 minutes)
python3 src/model_training_m1.py --max_train_samples 10000 --num_epochs 1

# Once you find good hyperparameters, scale up
# Full training overnight (8-12 hours)
python3 src/model_training_m1.py --num_epochs 3
```

### For Final Production Model:

**Use Colab with checkpointing:**
- Upload your optimized config
- Train full model (3-4 hours)
- Download the final checkpoint
- Use locally for inference

---

## Creating M1-Optimized Training Script

Let me create a training script optimized for your M1:

```python
# src/model_training_m1.py

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from evaluate import load
import numpy as np
import argparse

# Check for MPS (Apple Silicon)
if torch.backends.mps.is_available():
    device = "mps"
    print("✓ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    print("✓ Using CUDA GPU")
else:
    device = "cpu"
    print("⚠ Using CPU (will be slow)")

def main(args):
    # Load tokenizer and model
    print("\nLoading FLAN-T5-base...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    # Load datasets
    print("Loading datasets...")
    train_dataset = load_dataset('json', data_files='data/processed/train.jsonl', split='train')
    val_dataset = load_dataset('json', data_files='data/processed/val.jsonl', split='train')

    # Limit dataset size for quick experiments
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
        print(f"Using {args.max_train_samples} training samples for quick iteration")

    # Tokenization
    def preprocess_function(examples):
        inputs = tokenizer(
            examples['input'],
            max_length=args.max_input_length,
            truncation=True,
            padding='max_length'
        )
        targets = tokenizer(
            examples['target'],
            max_length=args.max_target_length,
            truncation=True,
            padding='max_length'
        )
        inputs['labels'] = targets['input_ids']
        return inputs

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names
    )

    # Metrics
    rouge = load('rouge')
    bleu = load('bleu')

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels)
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

    # Training arguments (M1-optimized)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,

        # Batch sizes optimized for 16GB M1
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Learning
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type='linear',
        warmup_steps=500,
        weight_decay=0.01,

        # MPS doesn't support fp16 yet
        fp16=False,

        # Evaluation and saving
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='rougeL',

        # Logging
        logging_steps=100,
        logging_dir=f'{args.output_dir}/logs',
        report_to='none',

        # Generation
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train
    print("\nStarting training...")
    print(f"Training samples: {len(tokenized_train)}")
    print(f"Validation samples: {len(tokenized_val)}")
    print(f"Device: {device}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")

    # Resume from checkpoint if exists
    last_checkpoint = None
    if args.resume_from_checkpoint:
        last_checkpoint = args.resume_from_checkpoint

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    print("\n✓ Training complete!")
    print(f"Model saved to: {args.output_dir}")

    # Final evaluation
    print("\nFinal evaluation...")
    metrics = trainer.evaluate()
    print(f"\nROUGE-L: {metrics['eval_rougeL']:.4f}")
    print(f"BLEU: {metrics['eval_bleu']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Limit training samples for quick iteration")
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=256)

    # Training
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Per-device batch size (4 works well on M1 Pro 16GB)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)

    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="./models/flan-t5-flashcard-v1")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()
    main(args)
```

---

## Quick Start Commands

### Test Run (10 minutes):
```bash
# Quick sanity check with small dataset
python3 src/model_training_m1.py \
    --max_train_samples 1000 \
    --num_epochs 1 \
    --batch_size 4
```

### Full Training (overnight):
```bash
# Start before bed, wake up to trained model
nohup python3 src/model_training_m1.py \
    --num_epochs 3 \
    --batch_size 4 \
    > training.log 2>&1 &

# Check progress
tail -f training.log
```

### Resume After Interruption:
```bash
# If training stopped, resume from last checkpoint
python3 src/model_training_m1.py \
    --resume_from_checkpoint ./models/flan-t5-flashcard-v1/checkpoint-3000
```

---

## My Recommendation

**For you specifically (M1 Pro, learning ML):**

1. **Start locally for iteration:**
   - Train on 10K samples (20 min)
   - Experiment with hyperparameters
   - Learn by watching it train
   - No Colab tokens used

2. **Once confident, do full run:**
   - Train overnight on M1 (8-12 hours)
   - Or use Colab for faster result (3-4 hours)

3. **Best of both:**
   - Develop and iterate locally
   - Final production model on Colab
   - Use Colab Pro only if you train a lot

**Your M1 Pro is perfect for this!** 💪
