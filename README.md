# FlashcardMaker - ML-Powered Flashcard Generation

Automatically generate high-quality flashcards from PDFs using a fine-tuned FLAN-T5 model.

## 🎓 Learning Path (New to ML?)

**This project is designed to teach you machine learning by doing!**

If you want to understand how everything works and build it yourself:

1. **Start Here:** [`LEARNING_GUIDE.md`](LEARNING_GUIDE.md) - Comprehensive guide explaining all ML concepts
2. **Hands-On:** [`notebooks/00_hands_on_exercises.ipynb`](notebooks/00_hands_on_exercises.ipynb) - Build key components yourself
3. **Tutorial Code:** [`src/data_preparation_tutorial.py`](src/data_preparation_tutorial.py) - Extensively commented educational version
4. **Debugging:** [`DEBUGGING_GUIDE.md`](DEBUGGING_GUIDE.md) - Common mistakes and solutions

**Learning Objectives:**
- Understand transformer models (FLAN-T5)
- Learn fine-tuning vs training from scratch
- Practice data preprocessing for NLP
- Master tokenization and seq2seq models
- Implement training loops and evaluation
- Debug real ML issues

**Just want to use it?** Skip to [Quick Start](#quick-start) below.

---

## Overview

This project fine-tunes FLAN-T5-base on 100K flashcard examples to generate question-answer pairs from educational text. The system can process PDFs, extract content, and produce study flashcards automatically.

**Key Features:**
- Fine-tuned FLAN-T5-base model for flashcard generation
- PDF text extraction and intelligent chunking
- Automatic Q&A generation from paragraphs
- Outputs in JSON and CSV formats
- Evaluation metrics (ROUGE, BLEU) and manual review tools
- **Educational resources for learning ML from scratch**

## Project Structure

```
FlashcardMaker/
├── data/
│   ├── raw/                    # Original 300K flashcard dataset (3 CSVs)
│   └── processed/              # Train/val/test splits (JSONL format)
├── src/
│   ├── data_preparation.py    # Process 300K CSVs → training format
│   ├── pdf_processor.py       # Extract & chunk PDF text
│   ├── flashcard_generator.py # Inference pipeline
│   └── evaluation.py          # Metrics & quality assessment
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Explore dataset
│   └── 02_baseline_training.ipynb   # Training on Google Colab
├── models/
│   └── flan-t5-flashcard-v1/        # Fine-tuned model checkpoints
├── outputs/                           # Generated flashcards
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Training Data

```bash
python src/data_preparation.py
```

This will:
- Load 300K flashcards from CSV files
- Filter to English examples (~100K)
- Create 80/10/10 train/val/test splits
- Save as JSONL files in `data/processed/`

### 3. Train the Model

**Option A: On Your M1 Mac (Recommended for Learning)**
```bash
# Quick test (10-20 minutes)
python3 src/model_training_m1.py --max_train_samples 1000 --num_epochs 1

# Full training overnight (8-12 hours)
nohup python3 src/model_training_m1.py --num_epochs 3 > training.log 2>&1 &
tail -f training.log  # Monitor progress
```

**Option B: Google Colab (Faster, 3-4 hours)**
1. Open `notebooks/02_baseline_training.ipynb` in Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Upload your data files to Google Drive
4. Run all cells

See [`COLAB_TIPS.md`](COLAB_TIPS.md) for handling disconnections and [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) for all commands.

### 4. Generate Flashcards from PDFs

```bash
python src/flashcard_generator.py models/flan-t5-flashcard-v1 your_document.pdf json
```

This will:
- Extract and chunk text from the PDF
- Generate flashcards for each chunk
- Save output to `outputs/flashcards_your_document.json`

## Usage Examples

### Data Exploration

```python
# Run the exploration notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

### PDF Processing

```python
from src.pdf_processor import process_pdf_to_chunks

# Extract chunks from a PDF
chunks = process_pdf_to_chunks('document.pdf', min_words=30, max_words=200)

print(f"Extracted {len(chunks)} chunks")
```

### Flashcard Generation

```python
from src.flashcard_generator import FlashcardGenerator

# Load the fine-tuned model
generator = FlashcardGenerator('models/flan-t5-flashcard-v1')

# Generate flashcards from a PDF
output_path = generator.process_pdf_to_flashcards(
    'document.pdf',
    output_format='json'  # or 'csv'
)

print(f"Flashcards saved to {output_path}")
```

### Evaluation

```python
from src.evaluation import FlashcardEvaluator

evaluator = FlashcardEvaluator()

# Evaluate on test set
metrics = evaluator.evaluate_from_jsonl(
    'predictions.jsonl',
    'data/processed/test.jsonl'
)

evaluator.print_metrics(metrics)

# Create manual review template
with open('outputs/flashcards.json') as f:
    flashcards = json.load(f)

evaluator.create_manual_review_template(flashcards)
```

## Performance Metrics

**Target Baseline Performance:**
- ROUGE-L > 0.35 (word/phrase overlap)
- BLEU > 0.25 (n-gram precision)
- 60%+ usable flashcards (manual review)

**Expected Results:**
- Training time: ~3-4 hours on Colab T4 GPU
- Inference speed: ~100 flashcards/minute on GPU
- Model size: ~1GB

## Model Details

**Base Model:** `google/flan-t5-base` (250M parameters)

**Training Configuration:**
- Batch size: 32 (8 per device × 4 gradient accumulation)
- Learning rate: 5e-5
- Epochs: 3
- Training examples: ~80K English flashcards

**Input Format:**
```
source_text: "Continuity is a key concept in Calculus..."
```

**Output Format:**
```
Q: What is Continuity in Calculus? A: Continuity is a fundamental concept...
```

## Data Preparation Details

The training data comes from a 300K flashcard dataset with:
- **Subjects:** Math, CS, Physics, Biology, Chemistry
- **Languages:** English, Urdu, Hindi (we use English only for baseline)
- **Metadata:** Difficulty, Bloom's taxonomy level, cognitive skills
- **License:** CC0-1.0 (Public Domain)

## Development Roadmap

### Phase 1: Baseline (Current)
- ✅ Data preparation pipeline
- ✅ FLAN-T5 training notebook
- ✅ PDF processing
- ✅ Basic inference
- ✅ Evaluation metrics

### Phase 2: Quality Improvements
- [ ] Add instruction prefixes to inputs
- [ ] Use full multilingual dataset (300K)
- [ ] Train for more epochs
- [ ] Experiment with generation parameters
- [ ] Quality filtering for outputs

### Phase 3: Enhanced Features
- [ ] Generate metadata (difficulty, bloom level)
- [ ] Multiple question types
- [ ] Topic-aware chunking
- [ ] Grammar checking
- [ ] A/B testing framework

### Phase 4: Deployment
- [ ] Model quantization (4-bit)
- [ ] REST API endpoint
- [ ] Web interface
- [ ] Batch processing system
- [ ] Cloud deployment

## Tips for Better Results

1. **PDF Quality Matters**
   - Clean, well-formatted PDFs work best
   - Multi-column layouts may scramble text
   - OCR may be needed for scanned documents

2. **Chunking Strategy**
   - Default: 30-200 words per chunk
   - Adjust based on your content density
   - Paragraphs work better than arbitrary splits

3. **Model Tuning**
   - Increase `num_beams` for better quality (slower)
   - Adjust `temperature` for more/less creativity
   - Try different `max_length` if answers are cut off

4. **Evaluation**
   - ROUGE/BLEU are useful but not perfect
   - Manual review is essential for quality
   - Would YOU use these flashcards to study?

## Troubleshooting

### Out of Memory During Training
```python
# Reduce batch size
per_device_train_batch_size = 4  # instead of 8
gradient_accumulation_steps = 8  # instead of 4
```

### Generated Text is Repetitive
```python
# Adjust generation parameters
temperature = 0.9
do_sample = True
top_p = 0.9
```

### PDF Extraction Issues
```python
# Try PyPDF2 if pdfplumber fails
from PyPDF2 import PdfReader
# Or preprocess PDFs with OCR tools
```

## Contributing

This is a personal project, but suggestions and improvements are welcome!

## License

MIT License - See LICENSE file for details

The training dataset is CC0-1.0 (Public Domain).

## Acknowledgments

- FLAN-T5 model: Google Research
- HuggingFace Transformers: Comprehensive ML library
- Flashcard dataset: Synthetically generated educational content

## Citation

If you use this project in your research, please cite:

```bibtex
@software{flashcardmaker2025,
  title={FlashcardMaker: ML-Powered Flashcard Generation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/FlashcardMaker}
}
```

---

**Questions or Issues?** Open an issue or reach out!

Happy learning! 📚
# FlashcardModel
