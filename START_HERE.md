# 🚀 START HERE - Your Learning Journey

Welcome to the ML-Powered Flashcard Generator! This is a real-world machine learning project designed specifically for learning.

## Two Paths: Choose Your Adventure

### Path 1: Learning Mode (Recommended for ML Students) 🎓

**Goal:** Understand machine learning by building this project step-by-step.

**You should choose this if:**
- You're new to machine learning
- You want to understand HOW and WHY things work
- You're willing to spend time learning concepts
- You want hands-on experience debugging and experimenting

**Your Journey:**

#### Week 1: Foundations
1. **Day 1-2: Understand the Concepts**
   - Read [`LEARNING_GUIDE.md`](LEARNING_GUIDE.md) - Sections 1-3
   - Focus on: What is FLAN-T5? What is fine-tuning? What is tokenization?
   - Watch recommended videos on transformers

2. **Day 3-4: Hands-On Exercises**
   - Open [`notebooks/00_hands_on_exercises.ipynb`](notebooks/00_hands_on_exercises.ipynb)
   - Complete Exercises 1-4 (data loading, filtering, transformation)
   - Don't skip! Type the code yourself

3. **Day 5-7: Data Pipeline**
   - Study [`src/data_preparation_tutorial.py`](src/data_preparation_tutorial.py)
   - Read ALL the comments - they explain WHY
   - Run the script: `python3 src/data_preparation_tutorial.py`
   - Inspect the output files

#### Week 2: Training
1. **Day 1-2: Understand Training**
   - Read [`LEARNING_GUIDE.md`](LEARNING_GUIDE.md) - Section 4
   - Learn: What happens during training? What is loss? What are gradients?

2. **Day 3-5: Train Your Model**
   - Set up Google Colab (free GPU!)
   - Open [`notebooks/02_baseline_training.ipynb`](notebooks/02_baseline_training.ipynb)
   - Read through EVERY cell before running
   - Start training (will take 3-4 hours)
   - Watch the loss decrease - you're teaching the model!

3. **Day 6-7: Evaluate Results**
   - Complete Exercises 5-7 in hands-on notebook
   - Understand ROUGE/BLEU metrics
   - Generate your first flashcards!

#### Week 3: Experiment & Iterate
1. **Experiment with variations:**
   - Try different learning rates
   - Modify the data format
   - Adjust generation parameters
   - Track what works and what doesn't

2. **Debug issues:**
   - Use [`DEBUGGING_GUIDE.md`](DEBUGGING_GUIDE.md) when stuck
   - Every error is a learning opportunity!

3. **Build something new:**
   - Add difficulty level generation
   - Try a different model (flan-t5-large)
   - Process your own PDFs

---

### Path 2: Quick Start Mode (Just Want Results) ⚡

**Goal:** Get a working flashcard generator as fast as possible.

**You should choose this if:**
- You're familiar with ML concepts
- You just want the tool working
- You'll learn by reading code later

**Steps:**

1. **Install (5 minutes)**
   ```bash
   cd /Users/adrianwill/Dev/FlashcardMaker
   pip install -r requirements.txt
   ```

2. **Prepare Data (10 minutes)**
   ```bash
   python3 src/data_preparation.py
   ```

3. **Train Model (3-4 hours on Google Colab)**
   - Upload `data/processed/*.jsonl` to Google Drive
   - Open `notebooks/02_baseline_training.ipynb` in Colab
   - Enable GPU, run all cells

4. **Generate Flashcards (5 minutes)**
   ```bash
   python3 src/flashcard_generator.py models/flan-t5-flashcard-v1 your.pdf json
   ```

Done! Read [`README.md`](README.md) for more details.

---

## What You're Building

### Input
A PDF containing educational content:
```
"Continuity is a key concept in Calculus within Mathematics.
It connects definitions with practical computations."
```

### Output
Flashcards in JSON format:
```json
{
  "question": "What is Continuity in Calculus?",
  "answer": "Continuity is a fundamental concept in Calculus that supports key methods and reasoning in Mathematics.",
  "source_text": "Continuity is a key concept..."
}
```

### How It Works
1. **Data Preparation:** Clean and format 100K training examples
2. **Fine-Tuning:** Teach FLAN-T5 model your flashcard style (3-4 hours)
3. **Inference:** Process new PDFs and generate flashcards (real-time)

---

## Key Resources

### Educational Materials
- **[`LEARNING_GUIDE.md`](LEARNING_GUIDE.md)** - Deep dive into every ML concept
- **[`notebooks/00_hands_on_exercises.ipynb`](notebooks/00_hands_on_exercises.ipynb)** - Learn by doing
- **[`src/data_preparation_tutorial.py`](src/data_preparation_tutorial.py)** - Commented code walkthrough
- **[`DEBUGGING_GUIDE.md`](DEBUGGING_GUIDE.md)** - Fix common issues

### Implementation Files
- **[`src/data_preparation.py`](src/data_preparation.py)** - Production data pipeline
- **[`src/pdf_processor.py`](src/pdf_processor.py)** - PDF text extraction
- **[`src/flashcard_generator.py`](src/flashcard_generator.py)** - Model inference
- **[`src/evaluation.py`](src/evaluation.py)** - Quality metrics

### Notebooks
- **[`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb)** - Explore dataset
- **[`notebooks/02_baseline_training.ipynb`](notebooks/02_baseline_training.ipynb)** - Train on Colab

---

## Common Questions

### "I'm stuck. Where do I get help?"

1. **Check DEBUGGING_GUIDE.md** - Covers 90% of common issues
2. **Re-read LEARNING_GUIDE.md** - Concepts you missed
3. **Google the error** - Most have been solved before
4. **HuggingFace Forums** - Active ML community
5. **Stack Overflow** - Tag [python][transformers]

### "What if I don't have a GPU?"

- **For training:** Use Google Colab (free T4 GPU)
- **For inference:** CPU works fine (just slower)

### "What if training doesn't work?"

- Start small: Train on just 1000 examples first
- Check data format: Print first 10 examples
- Lower learning rate: Try 1e-5 instead of 5e-5
- Read Section 5.3 in DEBUGGING_GUIDE.md

### "How long will this take to learn?"

**Realistic timeline:**
- **Week 1:** Understand concepts, run data prep
- **Week 2:** Train model, understand training
- **Week 3:** Experiment and iterate
- **Week 4+:** Build variations, improve quality

**Minimum to working system:** 3-4 days if you focus

### "What should I know beforehand?"

**Required:**
- Python basics (functions, classes, loops)
- Command line basics (running scripts)
- Basic math (what is a function, what is multiplication)

**Helpful but not required:**
- Pandas (we teach as we go)
- Neural networks (LEARNING_GUIDE explains)
- Transformers (we cover in detail)

---

## Success Criteria

### You'll know you're learning when:
- [ ] You can explain what tokenization does in your own words
- [ ] You understand why we split data into train/val/test
- [ ] You can debug a shape mismatch error
- [ ] You know why loss decreasing is good
- [ ] You can modify the code and see the effects

### You'll know you're done when:
- [ ] Model trains successfully (loss decreases)
- [ ] ROUGE-L score > 0.35 on test set
- [ ] Generated flashcards look reasonable (manual check)
- [ ] You can explain the full pipeline to someone else

---

## Final Tips

### For Learners:
1. **Don't rush** - Understanding > finishing fast
2. **Type, don't copy** - Muscle memory helps learning
3. **Break things** - Then fix them (best way to learn)
4. **Ask "why"** - For every line of code
5. **Take notes** - Keep a learning log

### For Quick Start Users:
1. **Read error messages** - They usually tell you exactly what's wrong
2. **Start small** - Test on 100 examples before full dataset
3. **Check GPU** - Training on CPU takes days, not hours
4. **Verify data** - Bad data = bad results

---

## What's Next?

Once you have the baseline working:

1. **Improve quality:**
   - Add metadata generation (difficulty, topics)
   - Use full 300K dataset (all languages)
   - Try larger model (flan-t5-large)

2. **Add features:**
   - Web interface for uploads
   - Batch processing multiple PDFs
   - Quality filtering

3. **Deploy:**
   - Create REST API
   - Host on cloud (AWS/GCP)
   - Share with others!

4. **Learn more:**
   - Read the FLAN-T5 paper
   - Try other models (BART, GPT)
   - Explore other NLP tasks

---

## Let's Begin!

**Ready to learn?** → Start with [`LEARNING_GUIDE.md`](LEARNING_GUIDE.md)

**Ready to build?** → Run `pip install -r requirements.txt`

**Questions?** → Check [`DEBUGGING_GUIDE.md`](DEBUGGING_GUIDE.md)

---

**Remember:** Machine learning is learned by doing, not by reading. Get your hands dirty, make mistakes, and have fun! 🎉

Good luck on your ML journey! 🚀
