# Progress Log

## 2026-04-24

Done:

- finalized project roadmap in [plan.md](/Users/adrianwill/Dev/convert/plan.md:1)
- created radiology manifest builder: [scripts/build_radiology_manifest.py](/Users/adrianwill/Dev/convert/scripts/build_radiology_manifest.py:1)
- generated [data/radiology_manifest.jsonl](/Users/adrianwill/Dev/convert/data/radiology_manifest.jsonl:1)
- manifest now contains `197` radiology rows
- manifest rows include `pdf_path`, `question`, `answer`, `page_number`, `slide_path`, and `status`
- completed image rendering script: [scripts/render_radiology_images.py](/Users/adrianwill/Dev/convert/scripts/render_radiology_images.py:1)
- generated `87` unique rendered slide PNGs in [data/slide_images](/Users/adrianwill/Dev/convert/data/slide_images)
- created manifest updater: [scripts/update_radiology_manifest.py](/Users/adrianwill/Dev/convert/scripts/update_radiology_manifest.py:1)
- created first-pass review script: [scripts/review_radiology_manifest.py](/Users/adrianwill/Dev/convert/scripts/review_radiology_manifest.py:1)
- completed first-pass manual review of manifest against rendered slide images
- added `status` to every manifest row with current counts:
  - `accept`: `173`
  - `edit`: `19`
  - `reject`: `5`
- used [data/sql/questions_rows.csv](/Users/adrianwill/Dev/convert/data/sql/questions_rows.csv:1) as extra signal to spot wording drift between original/current questions

Current Project State:

- scope: radiology-specific slice first
- task: `1 slide image -> 1 flashcard`
- milestone 1 progress:
  - manifest built
  - referenced slide images rendered
  - first-pass quality review completed
  - bad/awkward rows narrowed to `24` total (`19 edit + 5 reject`)
- main bottleneck now is cleanup of weak flashcards, not rendering or manifest plumbing

Suggested Next Steps:

- review only `edit` rows and rewrite them into clean standalone flashcards
- review `reject` rows and decide salvage vs permanent drop
- export cleaned subset into gold dataset file
- create held-out radiology split by `upload_id`
- run baseline model on held-out set for first eval

Open Questions:

- exact schema for final gold dataset export
- whether to keep `reject` rows in manifest or move them to separate review file
- exact format for train/val/test metadata
- which baseline model / prompt to use for first held-out eval
