# Quick Reference - Common Commands

## Training on Your M1 Mac

### Quick Test Run (10-20 minutes)
```bash
# Test everything works with small dataset
python3 src/model_training_m1.py \
    --max_train_samples 1000 \
    --num_epochs 1 \
    --batch_size 4
```

### Full Training (Overnight, 8-12 hours)
```bash
# Start before bed
nohup python3 src/model_training_m1.py \
    --num_epochs 3 \
    --batch_size 4 \
    > training.log 2>&1 &

# Check progress
tail -f training.log

# Or monitor with:
watch -n 10 'tail -20 training.log'
```

### Resume After Interruption
```bash
# Check what checkpoints exist
ls models/flan-t5-flashcard-v1/

# Resume from latest
python3 src/model_training_m1.py \
    --resume_from_checkpoint models/flan-t5-flashcard-v1/checkpoint-5000
```

---

## Training on Google Colab

### Setup
1. Go to https://colab.research.google.com/
2. Upload `notebooks/02_baseline_training.ipynb`
3. Runtime → Change runtime type → T4 GPU
4. Mount Google Drive
5. Upload `data/processed/*.jsonl` to Drive

### Run Training
- Just run all cells in order
- Takes 3-4 hours
- Download model from Drive when done

### If Disconnected
```python
# In Colab, check for checkpoints
!ls /content/drive/MyDrive/FlashcardMaker/models/flan-t5-flashcard-v1/

# Resume from checkpoint
trainer.train(resume_from_checkpoint=True)
```

---

## Data Preparation

### Prepare Training Data
```bash
# Run once before training
python3 src/data_preparation.py

# Output: data/processed/train.jsonl, val.jsonl, test.jsonl
```

### Explore Data
```bash
# Open data exploration notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## Generate Flashcards

### From Trained Model
```bash
# Generate flashcards from a PDF
python3 src/flashcard_generator.py \
    models/flan-t5-flashcard-v1 \
    your_document.pdf \
    json

# Output: outputs/flashcards_your_document.json
```

### Batch Process Multiple PDFs
```bash
# Put all PDFs in a folder
for pdf in data/pdf_inputs/*.pdf; do
    python3 src/flashcard_generator.py \
        models/flan-t5-flashcard-v1 \
        "$pdf" \
        json
done
```

---

## Evaluation

### Automatic Metrics
```bash
# Evaluate on test set
python3 src/evaluation.py evaluate \
    predictions.jsonl \
    data/processed/test.jsonl
```

### Manual Review
```bash
# Create review template
python3 src/evaluation.py review \
    outputs/flashcards_document.json

# Output: outputs/manual_review.csv
# Open in Excel/Google Sheets and score each flashcard
```

---

## Troubleshooting

### Check GPU/MPS Available
```bash
python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Out of Memory
```bash
# Reduce batch size
python3 src/model_training_m1.py \
    --batch_size 2 \
    --gradient_accumulation_steps 16
```

### Check Training Progress
```bash
# Monitor loss
grep "loss" training.log | tail -20

# Check validation metrics
grep "eval_rougeL" training.log
```

### List Checkpoints
```bash
# See saved checkpoints
ls -lh models/flan-t5-flashcard-v1/checkpoint-*

# Check sizes
du -sh models/flan-t5-flashcard-v1/*
```

---

## Common File Locations

```
FlashcardMaker/
├── data/
│   ├── raw/                           # Original CSV files
│   │   └── *.csv                      # 300K flashcards
│   └── processed/                     # Training data
│       ├── train.jsonl               # 80K examples
│       ├── val.jsonl                 # 10K examples
│       └── test.jsonl                # 10K examples
│
├── models/
│   └── flan-t5-flashcard-v1/         # Your trained model
│       ├── checkpoint-1000/          # Saved every 1000 steps
│       ├── checkpoint-2000/
│       ├── pytorch_model.bin         # Model weights
│       ├── config.json               # Model config
│       └── tokenizer/                # Tokenizer files
│
├── outputs/
│   ├── flashcards_*.json             # Generated flashcards
│   └── manual_review.csv             # For quality scoring
│
└── training.log                       # Training output (if using nohup)
```

---

## Hyperparameter Tuning

### Learning Rate
```bash
# Conservative (slower learning, more stable)
--learning_rate 1e-5

# Default (recommended)
--learning_rate 5e-5

# Aggressive (faster learning, less stable)
--learning_rate 1e-4
```

### Batch Size
```bash
# If OOM (out of memory)
--batch_size 2 --gradient_accumulation_steps 16

# Default for M1 16GB
--batch_size 4 --gradient_accumulation_steps 8

# If you have more memory
--batch_size 8 --gradient_accumulation_steps 4
```

### Epochs
```bash
# Quick test
--num_epochs 1

# Default
--num_epochs 3

# More training (watch for overfitting)
--num_epochs 5
```

---

## Experiment Tracking

### Create Experiment Log
```bash
# Create experiments directory
mkdir -p experiments

# Run experiment with descriptive name
python3 src/model_training_m1.py \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --output_dir experiments/lr_1e-4_epoch3 \
    > experiments/lr_1e-4_epoch3.log 2>&1

# Compare results
grep "eval_rougeL" experiments/*.log
```

---

## Daily Workflow

### Day 1: Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data
python3 src/data_preparation.py

# 3. Quick test
python3 src/model_training_m1.py --max_train_samples 1000 --num_epochs 1
```

### Day 2: Full Training
```bash
# Start overnight training
nohup python3 src/model_training_m1.py --num_epochs 3 > training.log 2>&1 &

# Next morning: check results
tail training.log
```

### Day 3: Generate & Evaluate
```bash
# Generate flashcards
python3 src/flashcard_generator.py models/flan-t5-flashcard-v1 test.pdf json

# Review quality
python3 src/evaluation.py review outputs/flashcards_test.json

# Open manual_review.csv and score samples
```

### Day 4+: Iterate
```bash
# Try different settings
python3 src/model_training_m1.py \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --output_dir models/experiment_2
```

---

## Get Help

- **Concept questions:** Read `LEARNING_GUIDE.md`
- **Errors:** Check `DEBUGGING_GUIDE.md`
- **Colab issues:** See `COLAB_TIPS.md`
- **Getting started:** Read `START_HERE.md`

---

## Keyboard Shortcuts

### Stop Training
- `Ctrl+C` (interrupt)
- Checkpoint will be saved automatically

### Background Process
- `Ctrl+Z` then `bg` (send to background)
- `jobs` (list background jobs)
- `fg %1` (bring job 1 to foreground)
- `kill %1` (kill job 1)

---

## Pro Tips

1. **Always start small:** Test with `--max_train_samples 1000` first
2. **Monitor temperature:** Keep your Mac cool, don't block vents
3. **Save often:** Use `--save_steps 500` for frequent checkpoints
4. **Log everything:** Use `nohup ... > training.log 2>&1 &`
5. **Version models:** Use descriptive output dirs like `models/v1-lr5e5-ep3`

---

**Happy training!** 🚀
