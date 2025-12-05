# ML Flashcard Generator - Learning Guide

**Welcome!** This guide will help you understand machine learning by building a real-world project: a flashcard generator powered by AI.

## What You'll Learn

By the end of this project, you'll understand:
1. How transformer models work (FLAN-T5)
2. Fine-tuning vs training from scratch
3. Data preprocessing for NLP
4. Tokenization and sequence-to-sequence models
5. Training loops, loss functions, and optimization
6. Evaluation metrics (ROUGE, BLEU)
7. Inference and generation strategies
8. Practical ML engineering (data pipelines, debugging, etc.)

---

## Learning Path (Start Here!)

### Phase 0: Prerequisites (Review if needed)

**Python Basics:**
- Functions, classes, dictionaries, lists
- File I/O (reading/writing files)
- Basic pandas for data manipulation

**ML Fundamentals:**
- What is supervised learning?
- What are neural networks?
- What is a loss function?
- What is backpropagation?

**Resources:**
- Python: [Official Python Tutorial](https://docs.python.org/3/tutorial/)
- ML Basics: [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- Deep Learning: [Fast.ai Practical Deep Learning](https://course.fast.ai/)

---

## Phase 1: Understanding the Task

### What Are We Building?

**Input:** A paragraph of text about a concept
```
"Continuity is a key concept in Calculus within Mathematics.
It connects definitions with practical computations."
```

**Output:** A question-answer flashcard
```
Q: What is Continuity in Calculus?
A: Continuity is a fundamental concept in Calculus that supports
   key methods and reasoning in Mathematics.
```

### Why Is This Hard?

1. **Understanding:** The model must understand the input text
2. **Reasoning:** It must identify what's important
3. **Generation:** It must create coherent questions and answers
4. **Structure:** It must follow the Q:/A: format

This is a **sequence-to-sequence (seq2seq)** task: transform one text sequence into another.

---

## Phase 2: Key ML Concepts Explained

### 2.1 What is FLAN-T5?

**T5 (Text-to-Text Transfer Transformer):**
- A model that treats ALL NLP tasks as text transformation
- Example: "Translate to French: Hello" → "Bonjour"
- Example: "Summarize: [long text]" → "[short summary]"

**FLAN (Finetuned Language Net):**
- T5 that was further trained on 1000+ instruction-following tasks
- Better at understanding what you want it to do
- Perfect for our task: "Generate flashcard from this text"

**Why T5 Architecture?**
```
INPUT TEXT → [ENCODER] → Hidden Representations → [DECODER] → OUTPUT TEXT
```

- **Encoder:** Reads and understands the input
- **Decoder:** Generates output word-by-word

Think of it like:
- Encoder = comprehension (reading the paragraph)
- Decoder = writing (creating the Q&A)

### 2.2 What is Fine-Tuning?

**Training from Scratch:**
- Start with random weights
- Requires MILLIONS of examples
- Takes weeks on many GPUs
- Very expensive

**Fine-Tuning:**
- Start with pre-trained model (FLAN-T5 already knows language!)
- Adjust weights for YOUR specific task
- Requires THOUSANDS of examples
- Takes hours on one GPU
- Much cheaper

**Analogy:**
- Training from scratch = Learning to read from scratch AND learning biology
- Fine-tuning = You already know how to read, just need to learn biology

**What we're doing:**
- FLAN-T5 already understands language and Q&A
- We're teaching it our specific flashcard style
- Using 100K examples of our format

### 2.3 Tokenization (CRITICAL CONCEPT)

**The Problem:** Neural networks don't understand text, only numbers

**The Solution:** Convert text to numbers (tokens)

```python
Text:  "What is Calculus?"
           ↓ Tokenizer
Tokens: [363, 19, 26234, 58]
           ↓ Embedding layer
Vectors: [[0.2, -0.5, 0.1, ...],    # "What"
          [0.8, 0.3, -0.2, ...],     # "is"
          [-0.1, 0.9, 0.4, ...],     # "Calculus"
          [0.5, -0.1, 0.7, ...]]     # "?"
```

**Important Details:**
- Each token is a word piece (not always a full word)
- "Calculus" might be one token, "Antidisestablishmentarianism" might be 5
- Special tokens: `<pad>` (padding), `</s>` (end of sequence)
- Token IDs are consistent: "Calculus" always = 26234

**Why This Matters:**
- Models have a maximum token limit (512 for input, 256 for output)
- If text is too long, it gets truncated
- If too short, it gets padded
- Different models use different tokenizers (can't mix them!)

### 2.4 Training Loop Explained

Here's what happens during training:

```python
for epoch in range(num_epochs):              # Epoch = one pass through all data
    for batch in training_data:              # Batch = small group of examples

        # 1. FORWARD PASS
        inputs = tokenize(batch.inputs)      # Convert text to tokens
        predictions = model(inputs)          # Model makes predictions

        # 2. COMPUTE LOSS
        loss = compare(predictions, batch.targets)  # How wrong were we?

        # 3. BACKWARD PASS (Backpropagation)
        gradients = compute_gradients(loss)  # Which direction to adjust weights?

        # 4. UPDATE WEIGHTS
        optimizer.step(gradients)            # Adjust weights to reduce loss

        # 5. REPEAT
        # Over time, loss decreases, model gets better!
```

**Key Concepts:**

**Loss Function:**
- Measures how wrong the model's predictions are
- For text generation: Cross-Entropy Loss
- Compares predicted probability distribution to actual next word
- Lower loss = better predictions

**Optimizer (Adam):**
- Algorithm that decides HOW to update weights
- Uses gradients (slopes) to find direction
- Uses learning rate to decide step size
- Adam is smart: adjusts learning rate automatically per parameter

**Learning Rate:**
- How big are the weight updates?
- Too high: Model jumps around, never converges
- Too low: Training takes forever
- We use 5e-5 (0.00005) - found through experimentation

**Batch Size:**
- How many examples to process before updating weights
- Larger = more stable, needs more memory
- Smaller = less stable, but fits in memory
- We use 32 (effective, through gradient accumulation)

**Gradient Accumulation:**
- Trick to simulate larger batches
- Process 8 examples, save gradients
- Process 8 more, add to gradients
- Do this 4 times (32 total)
- Then update weights
- Uses less memory than processing 32 at once!

### 2.5 Generation Strategies

When generating text, the model predicts one word at a time:

```
Input: "Continuity is a key concept..."

Step 1: Model predicts "Q:"
Step 2: Given "Q:", predicts "What"
Step 3: Given "Q: What", predicts "is"
Step 4: Given "Q: What is", predicts "Continuity"
... continues until </s> (end token)
```

**Greedy Decoding:**
- Always pick most likely next word
- Fast but can be boring/repetitive

**Beam Search:**
- Keep track of top N sequences (beams)
- Explore multiple possibilities
- Pick best overall sequence
- Better quality, slower
- We use beam_size=4

**Sampling:**
- Randomly pick next word based on probabilities
- More creative, but can be incoherent
- We DON'T use this (we want consistency)

**Parameters:**
- `temperature`: Higher = more random/creative (we use 0.7)
- `top_p`: Nucleus sampling, only sample from top P% probability mass
- `no_repeat_ngram_size`: Prevent repeating phrases (we use 3)

### 2.6 Evaluation Metrics

**Problem:** How do we measure if generated flashcards are good?

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**
- Measures word overlap between generated and reference text
- ROUGE-1: Individual word overlap
- ROUGE-2: Two-word phrase overlap
- ROUGE-L: Longest common subsequence

Example:
```
Reference: "What is Calculus in Mathematics?"
Generated: "What is Calculus?"

ROUGE-1: 3/4 = 0.75 (3 words match: What, is, Calculus)
ROUGE-L: 3/4 = 0.75 (longest match: "What is Calculus")
```

**BLEU (Bilingual Evaluation Understudy):**
- Originally for machine translation
- Measures n-gram precision (how many generated words are in reference)
- Includes brevity penalty (don't just generate one word!)

**Limitations:**
- Metrics don't capture meaning perfectly
- "The cat sat" vs "The feline sat" = low score but same meaning
- Manual review is essential!

**Our Targets:**
- ROUGE-L > 0.35 = decent baseline
- ROUGE-L > 0.5 = excellent
- 60%+ human-rated "good quality"

---

## Phase 3: Data Pipeline Deep Dive

### 3.1 Why Data Preprocessing Matters

**"Garbage In, Garbage Out"**
- ML models learn patterns from data
- Bad data = bad model, no matter how fancy the architecture
- Clean, consistent data = better results

### 3.2 Data Format Transformation

**What We Have (CSV rows):**
```python
{
    'source_text': 'Continuity is a key concept in Calculus...',
    'question': 'What is Continuity in Calculus?',
    'answer': 'Continuity is a fundamental concept...',
    'difficulty': 'hard',
    'bloom_level': 'Understand',
    # ... other metadata
}
```

**What We Need (Training pairs):**
```python
{
    'input': 'Continuity is a key concept in Calculus...',
    'target': 'Q: What is Continuity in Calculus? A: Continuity is a fundamental...'
}
```

**Why This Format?**
1. **Input:** Just the source text (what model sees)
2. **Target:** Q+A combined (what model should generate)
3. **Simple:** Model learns to map input → target
4. **Structured:** "Q: ... A: ..." helps model learn format

### 3.3 Train/Val/Test Splits

**Why Split?**

**Training Set (80%):**
- Model learns from this
- Sees these examples during training
- Updates weights based on these

**Validation Set (10%):**
- Model NEVER trains on this
- Used to check progress during training
- Early stopping: stop if validation gets worse
- Hyperparameter tuning: pick best learning rate, etc.

**Test Set (10%):**
- COMPLETELY held out until the very end
- Final evaluation of model performance
- Simulates real-world usage
- If test performance is much worse than validation, you overfit!

**Why Not Just Train/Test?**
- You'd tune hyperparameters on test set
- Test performance would be optimistically biased
- Validation is your "practice test"

**Stratification:**
- Ensure splits have similar distributions
- If 30% of data is Math, each split should be ~30% Math
- Prevents learning bias

---

## Phase 4: Understanding the Code

### 4.1 Data Preparation (`src/data_preparation.py`)

**Key Functions:**

```python
def load_all_data():
    # Why: Need to combine 3 CSV files into one DataFrame
    # How: pd.concat() stacks DataFrames vertically
    # Watch out: Make sure columns align!
```

```python
def filter_english():
    # Why: Start simple (one language), can expand later
    # How: df[df['language'] == 'en'] filters rows
    # Learn: Boolean indexing in pandas
```

```python
def format_for_training():
    # Why: Transform CSV format → model training format
    # How: Extract source_text, question, answer; combine Q+A
    # Key: This is YOUR design choice for the task!
```

```python
def create_train_val_test_split():
    # Why: Need separate data for train/validate/test
    # How: sklearn's train_test_split (with stratification)
    # Learn: Always use random_state for reproducibility
```

**Exercise:** Modify `format_for_training()` to include difficulty:
```python
target = f"[{difficulty.upper()}] Q: {question} A: {answer}"
# Model will learn to generate difficulty too!
```

### 4.2 PDF Processing (`src/pdf_processor.py`)

**Challenge:** PDFs are messy!
- Multi-column layouts can scramble text
- Headers/footers repeat on every page
- Tables, images, equations complicate extraction

**Our Strategy:**
1. Extract all text with pdfplumber (handles layout better)
2. Clean: remove extra whitespace, page numbers
3. Chunk: split into paragraphs (30-200 words)
4. Why 30-200? Matches training data length!

**Key Insight:**
- If chunks are too short: Not enough context for a good flashcard
- If chunks are too long: Multiple concepts, model gets confused
- Sweet spot: One paragraph = one concept = one flashcard

**Exercise:** Test chunking on different PDF types:
- Textbook (clean paragraphs)
- Research paper (dense text)
- Slides (bullet points)
Adjust `min_words` and `max_words` for each!

### 4.3 Model Training (Notebook)

**The HuggingFace Trainer API:**
- Abstracts away training loop complexity
- Handles: data loading, batching, GPU memory, checkpointing
- You just configure: batch size, learning rate, epochs

**What's Happening Under the Hood:**

```python
# Simplified pseudo-code of what Trainer does

for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Move to GPU
        batch = batch.to(device)

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()  # Compute gradients

        # Update weights
        if step % gradient_accumulation_steps == 0:
            optimizer.step()    # Update with accumulated gradients
            optimizer.zero_grad()  # Reset gradients

        # Log
        if step % logging_steps == 0:
            print(f"Loss: {loss.item()}")

        # Evaluate
        if step % eval_steps == 0:
            eval_loss = evaluate(model, val_dataloader)
            if eval_loss < best_loss:
                save_checkpoint(model)
```

**Hyperparameters Explained:**

```python
per_device_train_batch_size = 8
# How many examples fit in GPU memory at once
# Larger = faster BUT needs more memory
# If you get OOM (out of memory), reduce this

gradient_accumulation_steps = 4
# How many batches to accumulate before updating
# Effective batch size = 8 * 4 = 32
# Simulates larger batch with less memory

learning_rate = 5e-5
# How big are weight updates?
# For fine-tuning, use SMALL learning rates (don't break pretrained knowledge)
# Typical range: 1e-5 to 1e-4

num_train_epochs = 3
# How many times to see the entire training set
# More epochs = more learning BUT risk of overfitting
# Watch validation loss: if it increases, you're overfitting!

warmup_steps = 500
# Gradually increase learning rate at start
# Prevents big, random updates early on
# Stabilizes training
```

### 4.4 Generation (`src/flashcard_generator.py`)

**Key Concept:** Generation is autoregressive
- Model generates one token at a time
- Each token depends on all previous tokens
- Stops when model outputs `</s>` (end token) or reaches max_length

```python
# Pseudo-code of generation

output_tokens = []
current_input = input_tokens  # "Continuity is a key concept..."

while len(output_tokens) < max_length:
    # Predict next token
    logits = model(current_input)  # Probability distribution over vocab
    next_token = select_token(logits, strategy='beam_search')

    # Add to output
    output_tokens.append(next_token)

    # Update input (include generated tokens)
    current_input = concat(input_tokens, output_tokens)

    # Stop if end token
    if next_token == </s>:
        break

return decode(output_tokens)  # "Q: What is Continuity? A: ..."
```

**Beam Search Explained:**

Instead of just picking the most likely word, keep track of multiple possibilities:

```
Input: "Continuity is a key concept..."

Beam 1: "Q: What"       (prob: 0.8)
Beam 2: "Q: Explain"    (prob: 0.6)
Beam 3: "Question:"     (prob: 0.4)
Beam 4: "Q: Define"     (prob: 0.3)

For each beam, predict next word:
Beam 1 + "is" → "Q: What is" (prob: 0.8 * 0.9 = 0.72)
Beam 1 + "are" → "Q: What are" (prob: 0.8 * 0.3 = 0.24)
... continue for all beams

Keep top 4 sequences, repeat until done

Pick sequence with highest overall probability
```

Why beam search?
- Greedy (always pick best) can paint itself into a corner
- Beam search explores multiple paths
- Better overall quality

---

## Phase 5: Common Pitfalls & Debugging

### 5.1 Shape Mismatches

**Error:** `RuntimeError: Expected tensor of size [8, 512], got [8, 256]`

**What it means:** Data dimensions don't match what model expects

**How to debug:**
```python
print(f"Input shape: {inputs['input_ids'].shape}")
print(f"Labels shape: {labels.shape}")
print(f"Model expects: input_ids [batch_size, max_input_length]")
```

**Common cause:** Tokenizer max_length different from model's expectation

### 5.2 Out of Memory (OOM)

**Error:** `CUDA out of memory`

**Solutions:**
1. Reduce `per_device_train_batch_size`
2. Reduce `max_input_length` or `max_target_length`
3. Use gradient checkpointing (trades compute for memory)
4. Use smaller model (flan-t5-small instead of flan-t5-base)

### 5.3 Loss Not Decreasing

**Symptom:** Training loss stays high or fluctuates wildly

**Possible causes:**
1. **Learning rate too high:** Try 1e-5 instead of 5e-5
2. **Bad data:** Check for corrupted examples, weird characters
3. **Gradient explosion:** Enable gradient clipping
4. **Wrong loss function:** Make sure labels are set correctly

**Debug:**
```python
# Print a few training examples
for i in range(3):
    example = train_dataset[i]
    print(f"Input: {example['input']}")
    print(f"Target: {example['target']}")
    # Make sure they look right!
```

### 5.4 Good Training Loss, Bad Validation Loss

**Symptom:** Training loss decreases, validation loss increases

**Diagnosis:** OVERFITTING
- Model memorizes training data
- Doesn't generalize to new examples

**Solutions:**
1. More training data
2. Add regularization (weight decay, dropout)
3. Early stopping (stop when validation gets worse)
4. Simpler model
5. Data augmentation

### 5.5 Model Outputs Gibberish

**Symptom:** Generated text is random words or repetitive

**Possible causes:**
1. **Model not trained enough:** Train longer
2. **Generation params wrong:** Try different temperature, beam size
3. **Tokenizer mismatch:** Using wrong tokenizer for model
4. **Model never learned:** Check training loss - did it decrease?

**Debug:**
```python
# Generate with different parameters
gen1 = generate(text, temperature=0.5, num_beams=1)  # Greedy
gen2 = generate(text, temperature=0.7, num_beams=4)  # Beam search
gen3 = generate(text, temperature=1.0, num_beams=4, do_sample=True)  # Sampling

# Compare outputs
```

---

## Phase 6: Hands-On Exercises

### Exercise 1: Data Exploration
1. Load the flashcard dataset
2. Answer these questions:
   - What's the average source text length?
   - Which subject has the most examples?
   - Are questions shorter than answers? Why?
3. Find the longest source text - would it fit in 512 tokens?

### Exercise 2: Custom Data Format
Modify the training format to include difficulty:
```python
# Before
target = f"Q: {question} A: {answer}"

# After
target = f"[DIFFICULTY: {difficulty}] Q: {question} A: {answer}"
```
Retrain and see if model learns to generate difficulty levels!

### Exercise 3: Ablation Study
Train 3 models with different learning rates:
- 1e-5 (conservative)
- 5e-5 (our default)
- 1e-4 (aggressive)

Compare validation loss curves. Which is best? Why?

### Exercise 4: Generation Experiments
Take one input text, generate flashcards with different settings:
```python
# Conservative
generate(text, temperature=0.5, num_beams=8)

# Balanced
generate(text, temperature=0.7, num_beams=4)

# Creative
generate(text, temperature=1.0, num_beams=2)
```
Which produces best flashcards? Why?

### Exercise 5: Error Analysis
1. Generate 100 flashcards from test set
2. Manually identify 10 worst ones
3. Categorize errors:
   - Wrong question
   - Wrong answer
   - Format error
   - Irrelevant
4. What pattern do you see? How could you fix it?

---

## Phase 7: Going Deeper

### Advanced Topics to Explore:

**1. Attention Mechanisms**
- How does the model know which part of input to focus on?
- Visualize attention weights during generation
- See which source words influence which generated words

**2. Transfer Learning Theory**
- Why does fine-tuning work so well?
- What does "pretrained knowledge" actually mean?
- When would you train from scratch instead?

**3. Scaling Laws**
- How does model size affect performance?
- Try flan-t5-small (60M), flan-t5-base (250M), flan-t5-large (780M)
- Plot performance vs size

**4. Data Efficiency**
- How much data do you actually need?
- Train on 10K, 50K, 100K examples
- Plot learning curves

**5. Multi-task Learning**
- Can you train on flashcards AND summarization at the same time?
- Does it help or hurt?

---

## Resources for Further Learning

### Courses:
1. **Fast.ai Practical Deep Learning** - Learn by building
2. **Stanford CS224N** - NLP with Deep Learning
3. **HuggingFace Course** - Transformers in depth
4. **DeepLearning.AI** - Andrew Ng's specialization

### Papers to Read:
1. **"Attention Is All You Need"** - Original transformer paper
2. **"Exploring the Limits of Transfer Learning with T5"** - T5 architecture
3. **"Finetuned Language Models Are Zero-Shot Learners"** - FLAN training

### Tools:
1. **HuggingFace Hub** - Explore thousands of models
2. **Weights & Biases** - Experiment tracking
3. **TensorBoard** - Visualize training

### Communities:
1. **HuggingFace Forums** - Ask questions
2. **r/MachineLearning** - Stay updated
3. **Papers With Code** - Find implementations

---

## Final Thoughts

**Learning ML is like learning to cook:**
1. Start with recipes (tutorials) - follow exactly
2. Understand ingredients (concepts) - why each step?
3. Experiment (projects) - what if I change this?
4. Create your own (research) - new recipes entirely!

**You're at stage 1-2 with this project. That's perfect!**

The best way to learn is to:
1. **Run the code** - see it work
2. **Break the code** - see what fails
3. **Fix the code** - understand why
4. **Modify the code** - make it yours

**Don't just copy-paste. Type it out. Read every comment. Run experiments.**

Good luck, and have fun! ML is an amazing field. 🚀
