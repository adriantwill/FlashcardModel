# AI Flashcards & Notes Dataset (Synthetic) 

**Rows:** 300000  
**Files:** 3 CSV parts  
**Created:** 2025-11-02 10:22:46 UTC

## Description
This dataset contains synthetic educational content aligned to typical student learning tasks:
- Short **source_text** passages (lecture-note style)
- A concise **summary**
- A **question–answer (Q/A)** flashcard pair
- Labels for **subject, topic, subtopic, difficulty, Bloom’s level, cognitive skill, language**, and **source type**
- An approximate **token_estimate** for modeling/storage planning

Designed for:
- NLP: text summarization, Q/A generation, classification
- Recommendation: topic/syllabus tagging, difficulty estimation
- EdTech research and prototyping

## Files
- `ai_flashcards_notes_dataset_bisma_v1_part1.csv`
- `ai_flashcards_notes_dataset_bisma_v1_part2.csv`
- `ai_flashcards_notes_dataset_bisma_v1_part3.csv`

## Schema
| Column          | Type    | Description |
|-----------------|---------|-------------|
| id              | int     | Row identifier |
| subject         | string  | High-level subject area (Math, CS, Physics, Biology, Chemistry) |
| topic           | string  | Topic within the subject (e.g., Calculus, Algorithms) |
| subtopic        | string  | Specific concept (e.g., Derivatives, Hash Tables) |
| difficulty      | string  | One of: easy, medium, hard |
| language        | string  | 'en', 'ur', or 'hi' (synthetic mark) |
| bloom_level     | string  | Remember, Understand, Apply, Analyze, Evaluate, Create |
| cognitive_skill | string  | Definition, Comprehension, Computation, Comparison, Inference, Synthesis |
| source_type     | string  | lecture_note, textbook_excerpt, web_article, research_summary |
| source_text     | string  | Short paragraph explaining the subtopic |
| summary         | string  | One-line summary |
| question        | string  | Flashcard-style question |
| answer          | string  | Corresponding answer |
| token_estimate  | int     | Rough token count (~4 chars/token) |


