# AGENTS.md

## Purpose

This repo exists to build a **resume-grade applied AI project** and to help the user **learn by building**.

Primary reference: [plan.md](/Users/adrianwill/Dev/convert/plan.md:1)
Progress tracker: [progress.md](/Users/adrianwill/Dev/convert/progress.md:1)

Agents working in this repo must optimize for:

- strong learning value
- strong resume value
- clear technical understanding
- measurable progress

Do not optimize for speed alone if it reduces understanding or project quality.

## Mentorship Rule

Follow the mentorship rule from [plan.md](/Users/adrianwill/Dev/convert/plan.md:1):

- do not silently scaffold large systems the user will not understand
- do not push giant template-heavy solutions
- prefer guidance, small scripts, reviews, eval harnesses, and incremental implementation
- prefer code the user can explain end to end
- keep experiments interpretable

Success means user can clearly explain:

- data flow end to end
- dataset construction and labeling logic
- split strategy
- baseline and eval design
- why a modeling choice worked or failed

## Working Style

- Treat this as a **learning project first** and a **resume project second**, not a speedrun.
- Favor narrow, defensible milestones over flashy but shallow progress.
- Push toward evidence: dataset quality, eval results, measured improvements, and integration.
- Avoid premature fine-tuning, premature deployment, and premature infra complexity.

## Progress Discipline

At start of each work session:

- read [plan.md](/Users/adrianwill/Dev/convert/plan.md:1)
- read [progress.md](/Users/adrianwill/Dev/convert/progress.md:1)
- align new work with current milestone

At end of each meaningful work chunk:

- update [progress.md](/Users/adrianwill/Dev/convert/progress.md:1)
- record what changed, current counts/state, and next recommended step
- keep progress notes concise and factual

Do not let `progress.md` drift out of date.

## Current Project Direction

Current best path:

1. build trustworthy gold dataset
2. clean weak flashcards
3. split by lecture/upload
4. run baseline eval
5. fine-tune only after baseline and dataset quality are solid

Avoid:

- full-model training from scratch
- giant unreviewed code generation
- random-row splits
- training on noisy labels
- spending large effort on polish before evals exist

## Decision Standard

When choosing between two approaches, prefer one that better supports:

- user understanding
- resume credibility
- measurable evaluation
- maintainable incremental progress
