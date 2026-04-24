# Slide -> Q/A Model Plan

## Goal

Build resume-grade applied AI project:

- input: lecture slide image
- output: high-quality study question and answer
- integration target: `~/Dev/study-sesh`
- learning target: data pipeline, evals, multimodal fine-tuning, product integration, model serving
- working style target: **you write code yourself; plan and guidance should teach, not outsource implementation**

Project should prove more than "called model API." It should show:

- you designed dataset + labeling loop
- you built baseline + eval harness
- you fine-tuned model with measurable gains
- you shipped model into real app
- you understood why each modeling choice worked or failed

## Mentorship Rule

This plan is intentionally written for **learn-by-building**.

- do not ask AI to silently scaffold whole training stack
- do not copy giant repo template you do not understand
- do write each stage yourself after you can explain it plainly
- use guidance, pseudocode, reviews, and debugging help
- keep notebook experiments small and interpretable

Success means:

- you can explain data flow end to end
- you can explain loss, split strategy, evals, and failure modes
- you can defend why you chose fine-tuning over pure prompting

## Big Recommendation

Do **not** frame this as "train giant model from scratch."

Best applied-AI version:

1. build strong baseline from current pipeline
2. create high-quality supervised dataset from your real slides
3. fine-tune open vision-language model with LoRA/QLoRA
4. compare against baseline with real metrics + human review
5. deploy into `study-sesh` behind model router / A-B switch

That is better engineering, better ML, better resume story.

## Current Starting Point

`study-sesh` already has baseline pipeline in [question-generator.ts](/Users/adrianwill/Dev/study-sesh/src/lib/ai/question-generator.ts:1):

- PDF -> PNG with Poppler
- slide image -> Gemini prompt
- returns question/answer/options JSON

`convert` already looks like dataset workbench:

- PDFs in `supabase-files/`
- exported questions in `questions_rows.csv`
- upload metadata in `uploads_rows.csv`
- page-labeling notes in `pdf_question_labeling.md`

This is strong raw material for supervised fine-tuning and eval.

## Current Status

Completed so far:

- radiology manifest builder created: [build_radiology_manifest.py](/Users/adrianwill/Dev/convert/scripts/build_radiology_manifest.py:1)
- radiology manifest generated: [radiology_manifest.jsonl](/Users/adrianwill/Dev/convert/data/radiology_manifest.jsonl:1)
- radiology slide renderer created: [render_radiology_images.py](/Users/adrianwill/Dev/convert/scripts/render_radiology_images.py:1)
- referenced slide images generated in [slide_images](/Users/adrianwill/Dev/convert/data/slide_images)

Current focus should remain:

- gold dataset quality
- eval split
- baseline measurement

Not yet:

- fine-tuning
- deployment
- product integration of custom model

## Best Project Scope

Train **one narrow, useful model** first:

`slide image -> 1 high-yield flashcard pair`

Why this scope:

- simpler target than "generate full deck"
- easier to label well
- easier to evaluate
- easier to deploy with low latency
- stronger first resume project

Later expansion:

- multi-question generation per slide
- reasoning tags
- difficulty calibration
- confidence score
- page retrieval / slide alignment

## Model Recommendation

Start with **adapter fine-tune on open VLM**, not full fine-tune.

Recommended first path:

- base model: `Qwen/Qwen2.5-VL-3B-Instruct`
- training method: LoRA or QLoRA
- trainer stack: `transformers` + `trl` + `peft` + `accelerate`
- inference target: server-side job or hosted GPU endpoint

Why:

- strong image + text understanding
- realistic to fine-tune
- enough capacity for slide reading without 7B/72B cost
- easier story for portfolio than closed-model prompt tuning only

If quality ceiling too low, next step:

- `Qwen/Qwen2.5-VL-7B-Instruct`

Avoid first project traps:

- full-parameter fine-tune
- custom architecture from scratch
- direct RL loop before strong eval baseline
- trying to solve OCR, retrieval, deck generation, personalization, and tutoring all at once

## Budget Reality

Do **not** assume Colab Free is enough for clean iteration.

Possible:

- small experiments
- dataset debugging
- tiny proof-of-concept fine-tune

Not reliable for serious project:

- repeatable long runs
- stable GPU type
- larger VLM fine-tunes
- resume-grade experiment cadence

More realistic expectation:

- initial debugging on local/Colab
- first real LoRA training on rented GPU

Likely low-budget path:

- use free/local for data prep and eval code
- rent GPU only for actual training runs
- shut it down immediately after run

Rough practical expectation:

- prototype cost can be low
- full project probably not zero-dollar
- biggest cost driver is repeated experimentation, not one training run

Important:

Do not optimize for "free" so hard that project becomes weak. Optimize for **high learning per dollar**.

## Product Architecture

### V1

1. user uploads PDF in `study-sesh`
2. render slides to images
3. run baseline or fine-tuned model per slide
4. save generated Q/A rows to Supabase
5. let user edit / accept / reject
6. log edits as future training signal

### V2

1. slide -> candidate flashcard
2. confidence / quality filter decides keep or skip
3. human edits feed active-learning loop

Important design choice:

Keep **generation** and **filtering/ranking** separate in code, even if same model does both at first. Better for evals and future iteration.

## Data Plan

## Phase 1: Build gold dataset

Need dataset row like:

```json
{
  "image_path": "rendered/slide_014.png",
  "slide_id": "upload123_page14",
  "course": "GI physiology",
  "ocr_text": "...",
  "question": "What do Brunner's glands secrete?",
  "answer": "Alkaline mucus rich in bicarbonate",
  "page_number": 14,
  "quality_label": 1
}
```

Sources:

- existing PDFs
- existing question exports
- page-number matches
- your own manual edits in app

### Label quality bar

Keep only rows where:

- question answerable from one slide
- answer concise and unambiguous
- question tests high-yield concept
- no duplicate cards
- no trivial copy-paste wording unless term definition requires it

### Data volume target

- first serious run: 1,000 to 3,000 gold rows
- stretch goal: 5,000+ rows

That is enough for meaningful LoRA experiment if labels are clean.

### Fast labeling strategy

1. render all slides
2. align existing questions to pages
3. auto-propose `(slide, question)` matches
4. review in lightweight labeling UI or CSV workflow
5. mark `accept / edit / reject`
6. keep only accepted+edited rows for train set

### Data split

Split by **lecture / upload**, not random row.

Use:

- 70% train
- 15% validation
- 15% test

Reason:

Random split leaks near-duplicate slides and inflates score.

## Modeling Plan

## Stage 0: Baseline

Before training anything, benchmark:

- current Gemini pipeline
- optional OCR-only text baseline
- optional open-model zero-shot baseline

Store outputs on same test set.

## Stage 1: SFT for single-card generation

Train on chat-style multimodal examples:

- input: slide image + instruction
- output: strict JSON with `question`, `answer`, `options`

Prompt shape:

```text
Generate exactly 1 high-yield study flashcard from this lecture slide.
Return JSON only with:
question, answer, options
Rules:
- ask about most testable concept on slide
- answer must be concise
- 3 distractors, plausible but wrong
- skip if slide has no testable content
```

Training objective:

- supervised fine-tuning
- teacher forcing on gold JSON output

## Stage 2: Preference / ranking layer

After SFT works, add one:

- small reranker model, or
- pairwise preference dataset from accepted vs rejected outputs

This gives stronger "applied AI engineer" signal than only fine-tuning.

## Stage 3: Edit-learning loop

Log:

- original model output
- user final edited output
- accept/reject
- time to edit

Use edited cards as next training batch.

## Evaluation Plan

Need hard evals. Not "looks better."

### Automatic metrics

- exact / fuzzy answer overlap
- semantic similarity for answer
- JSON validity rate
- duplicate question rate
- skip precision on low-content slides

### Human rubric

Rate 1-5 on:

- correctness
- usefulness for exam study
- specificity
- concision

### Product metrics

- acceptance rate
- average edits per accepted card
- cards per minute
- latency per slide

### Win condition

Model project is strong if it beats baseline on:

- human quality score
- acceptance rate
- edit distance to final accepted card

## Deep ML Learning Track

If your goal is strong resume story **and** real ML depth, learn each layer in order:

### Layer 1: Problem framing

Learn:

- why narrow task definition matters
- why label quality dominates early results
- why lecture-level split matters

Deliverable:

- written task spec
- error taxonomy before training

### Layer 2: Data

Learn:

- supervised example design
- noisy-label cleanup
- train/val/test leakage
- class imbalance between content-heavy and low-content slides

Deliverable:

- dataset card
- reproducible dataset build script

### Layer 3: Training mechanics

Learn:

- tokenization of chat-style outputs
- teacher forcing
- cross-entropy loss
- LoRA intuition: low-rank update instead of full weight updates
- why QLoRA trades compute for approximation

Deliverable:

- short write-up of training objective in your own words
- one plotted training curve you can interpret

### Layer 4: Evaluation

Learn:

- why automatic metrics can mislead
- why human eval needed for educational quality
- how to compare baseline vs fine-tuned model honestly

Deliverable:

- eval script
- annotated failure examples

### Layer 5: Serving + product loop

Learn:

- batch vs online inference
- latency vs quality tradeoff
- logging model outputs for active learning

Deliverable:

- deployed inference path
- user-edit logging schema

## Training Stack

Use Python training repo or subdir, separate from app runtime.

Suggested stack:

- Python 3.11+
- `.venv`
- `torch`
- `transformers`
- `datasets`
- `trl`
- `peft`
- `accelerate`
- `wandb` or TensorBoard

Likely artifacts:

- `data/render_slides.py`
- `data/build_dataset.py`
- `data/train.jsonl`
- `data/val.jsonl`
- `train_vlm.py`
- `eval_vlm.py`
- `serve_vlm.py`
- `notebooks/error_analysis.ipynb`

Keep code simple on purpose:

- one script to build dataset
- one script to train
- one script to eval
- one inference adapter

If codebase starts feeling magical, simplify.

## Integration Into Study Sesh

Add model router in `study-sesh`:

- `gemini_baseline`
- `ft_qwen_local`
- `ft_qwen_remote`

Desired server contract:

```ts
type GeneratedCard = {
  question: string;
  answer: string;
  confidence?: number;
  model?: string;
};
```

Integration steps:

1. keep current Gemini path
2. add fine-tuned model adapter behind env flag
3. log outputs and user edits
4. expose side-by-side compare mode for you only
5. ship best route to default later

## Resume / Portfolio Angle

Make project easy to explain in 30 seconds:

> Built multimodal study-card generation system that converts lecture slide images into exam-style Q/A pairs. Created dataset pipeline from real PDFs, fine-tuned open VLM with LoRA, built eval harness against prompt-engineered baseline, and integrated model into production Next.js study app.

Artifacts that make project stand out:

- short demo video
- public architecture diagram
- eval dashboard screenshot
- write-up with failure cases and ablations
- before/after examples against Gemini baseline
- explanation section: "what I coded myself and what I intentionally did not abstract away"

## Learning Roadmap

This project teaches good applied-AI muscles:

- data curation
- multimodal dataset design
- train/val/test discipline
- PEFT / LoRA
- offline evals + human evals
- inference serving
- product instrumentation
- active learning loop

Math/ML topics worth learning in parallel:

- cross-entropy loss
- tokenization and causal LM training
- LoRA intuition
- embedding similarity
- calibration / confidence
- ranking losses
- train/val leakage
- distribution shift
- overfitting signals
- decoding tradeoffs

Recommended learning style:

- after each stage, write 5 to 10 bullet notes:
  - what you expected
  - what actually happened
  - what metric moved
  - what confused you
  - what you changed next

This reflection habit matters for interviews.

## 10-Week Execution Plan

### Week 1

- freeze target task: `1 slide -> 1 card`
- render slide images
- audit existing CSV/PDF data
- define labeling schema
- build held-out test split by lecture
- read enough to explain LoRA, SFT, cross-entropy, and lecture-level split

### Week 2

- align old questions to slide pages
- manually clean first 200 to 300 gold examples
- build baseline eval script
- score Gemini baseline on held-out set
- write first error taxonomy

### Week 3

- build training dataset export
- run zero-shot open-model baseline
- train first LoRA on small subset
- verify JSON formatting + inference script
- write short note: what SFT optimizing mathematically

### Week 4

- run error analysis
- fix data issues
- tune prompts, decoding, skip logic
- compare 3B vs baseline
- decide if dataset quality or model quality is current bottleneck

### Week 5

- scale dataset to 1k+ gold examples
- retrain best config
- add product logging for accept/edit/reject
- document train/val gap and overfitting signals

### Week 6

- integrate model router into `study-sesh`
- add private compare mode
- collect your own usage feedback
- analyze latency and server cost assumptions

### Week 7

- improve skip behavior on low-value slides
- reduce duplicates / vague questions
- create better human eval rubric

### Week 8

- run second serious training round
- compare against first round with fixed eval set
- write ablation notes

### Week 9

- final eval report
- clean architecture diagram
- demo video
- draft project write-up
- resume bullets

### Week 10

- polish repo/docs
- rehearse 2-minute explanation
- write "future work" grounded in actual failures

## Risks

### Risk: labels noisy

Fix:

- tighten quality bar
- keep smaller but cleaner dataset
- split by lecture, not row

### Risk: model hallucinates

Fix:

- require concise extractive-style answers
- add skip option
- use reranking / filtering

### Risk: GPU budget too high

Fix:

- start with 3B + QLoRA
- train only on single-card task
- use hosted GPU for training, CPU/API for baseline eval
- keep first runs tiny and diagnostic, not heroic

### Risk: output not better than Gemini

Fix:

- that is still valid if eval honest
- ship hybrid router
- show where fine-tuned model wins: cost, control, privacy, domain adaptation

## Concrete Success Criteria

Project is success if all true:

- reproducible dataset build pipeline
- held-out eval set with lecture-level split
- fine-tuned model beats or matches baseline on human quality
- integrated into `study-sesh` behind model flag
- write-up explains wins, failures, next steps
- you can whiteboard system from memory and explain each major tradeoff

## Questions To Answer Before Build

1. Can you publish small cleaned subset, or must all data stay private?
2. Do you want single-course prototype first, or many-course generalized model?
3. Do you want strongest first milestone to be training win, or app integration win?
4. How much manual labeling time can you realistically do each week?
5. Do you want to optimize first for quality, cost, or latency?

## Immediate Next Step

Do this first:

1. create gold test set from 10 to 15 full lecture PDFs
2. benchmark current Gemini pipeline on it
3. manually clean first 200 examples
4. only then start fine-tuning

Without this, training loop becomes guesswork.

## Milestone Structure

### Milestone 1

Radiology-specific **problem definition + gold dataset + baseline + eval harness**

End state:

- radiology-only scope selected
- exact task frozen: `1 slide -> 1 flashcard`
- lecture-level train/val/test split created
- 150 to 300 cleaned gold examples prepared
- Gemini baseline outputs saved on held-out set
- human eval rubric defined
- first error taxonomy written

This milestone does **not** require fine-tuning yet.

Reason:

- better ML learning
- avoids training on weak labels
- avoids fake progress from one-off runs
- makes later fine-tune results interpretable

### Milestone 2

First fine-tuned radiology-specific prototype model

End state:

- first LoRA/QLoRA training run completed
- inference path works on held-out slides
- outputs compared against baseline
- failure analysis written
- next experiment justified by evidence

## Useful Current Docs

- TRL SFT trainer: https://huggingface.co/docs/trl/en/sft_trainer
- VLM SFT guide: https://huggingface.co/docs/trl/main/en/training_vlm_sft
- Qwen2.5-VL model docs: https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl
- Qwen2.5-VL release notes: https://qwenlm.github.io/blog/qwen2.5-vl/
