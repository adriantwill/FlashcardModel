This portfolio project is a mid-level Python Machine Learning Engineering pipeline that fine-tunes the Qwen3-VL-8B-Instruct vision-language model on a custom business dataset of 5,000 images and 13,000 text question-answer pairs. To prove robust software and framework engineering skills to hiring managers, the project completely skips high-level "all-in-one" wrappers like Hugging Face's SFTTrainer. Instead, it uses core library building blocks (torch, transformers, and peft) to construct a fully custom, hand-coded architecture—including an explicit torch.utils.data.Dataset pipeline for multi-modal alignment, a manually configured 4-bit QLoRA adapter injection, a raw Python training loop with explicit backpropagation and native mixed precision (torch.amp), and a production-ready FastAPI inference endpoint.

> **Project status:** Planning and data preparation. Model features and dataset counts above are targets until the roadmap validates them.

## Development strategy

Use the Mac for data work, unit tests, evaluation code, API development, and small-model experiments. Use a Linux cloud machine with an NVIDIA GPU for QLoRA integration, training, and final evaluation. Mac 4-bit support exists but is slower; the cloud run is the source of truth for the CUDA-focused training stack.

The final target is [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct). Start with the official 2B or 4B checkpoint when testing the pipeline so mistakes are faster and cheaper to find.

## Learning roadmap

Complete one checkbox at a time. Each step ends with a result that can be tested before moving forward.

### Phase 1: Build and test locally on the Mac

- [ ] **1. Define the task and success criteria**
  - Learn: the difference between training goals, evaluation metrics, and product behavior.
  - Build: a short statement describing the input image, expected answer, target user, and project limitations.
  - Done when: five representative examples and their expected answers are written down. Because the data is medical education material, state that the project is not a clinical diagnostic tool.

- [ ] **2. Create the Python project environment**
  - Learn: virtual environments, dependency files, modules, and tests.
  - Build: a `.venv`, dependency file, source directory, test directory, and configuration file.
  - Done when: a clean virtual environment can install the project and run one test. Use `.venv` for every `pip` command.

- [ ] **3. Validate the dataset**
  - Learn: CSV parsing, file paths, image loading, and data-quality checks.
  - Build: a script that checks required columns, blank questions or answers, duplicate IDs, missing image paths, and unreadable images.
  - Done when: the script reports exact row and image counts and exits with no unexplained invalid records.

- [ ] **4. Explore and split the dataset**
  - Learn: data distributions and train/validation/test leakage.
  - Build: a small report showing answer lengths, image sizes, duplicate questions, and example records. Split by source document or slide deck rather than random rows when related slides could leak across splits.
  - Done when: train, validation, and test files have no source overlap and their counts are recorded.

- [ ] **5. Implement the multimodal dataset pipeline**
  - Learn: PyTorch [`Dataset` and `DataLoader`](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html), Qwen's [`AutoProcessor`](https://huggingface.co/docs/transformers/model_doc/qwen3_vl), and [multimodal chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating_multimodal).
  - Build: a custom dataset plus collator that loads an image and question, applies the chat template, tokenizes the answer, pads batches, and masks non-answer labels.
  - Done when: one batch has documented tensor shapes and dtypes, decodes back to the expected example, and passes unit tests.

- [ ] **6. Measure a pretrained baseline**
  - Learn: inference, prompt consistency, and why fine-tuning needs a baseline comparison.
  - Build: run the same frozen prompt and evaluation examples through Qwen3-VL-2B or Qwen3-VL-4B without training.
  - Done when: predictions and baseline metrics are saved in a reproducible file.

- [ ] **7. Build the evaluation harness**
  - Learn: exact-match metrics, normalized text comparison, and human review rubrics.
  - Build: automatic scoring plus a small error-analysis report for incorrect, incomplete, and unsupported answers.
  - Done when: one command evaluates both the pretrained model and future fine-tuned checkpoints on the untouched test set.

### Phase 2: Train on a cloud NVIDIA GPU

- [ ] **8. Reproduce the baseline in the cloud**
  - Learn: CUDA devices, GPU memory, reproducible environments, and experiment configuration.
  - Build: recreate the local baseline using pinned dependencies and the same examples on the cloud GPU.
  - Done when: the model loads, inference succeeds, GPU memory usage is recorded, and outputs are acceptably close to the local baseline.

- [ ] **9. Add 4-bit QLoRA manually**
  - Learn: [4-bit quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes) and [LoRA adapters](https://huggingface.co/docs/peft/main/conceptual_guides/lora).
  - Build: load the base model in 4-bit, freeze its weights, inject LoRA into selected modules, and print trainable parameter counts. Use the [official Qwen fine-tuning code](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune) as a reference while keeping this project's implementation explicit.
  - Done when: only intended adapter parameters require gradients and one forward/backward pass finishes without an out-of-memory error.

- [ ] **10. Write and verify the raw training loop**
  - Learn: loss, backpropagation, gradient accumulation, clipping, optimizer steps, schedulers, checkpointing, and [`torch.amp`](https://docs.pytorch.org/docs/stable/amp.html).
  - Build: a loop with mixed precision, logging, validation, checkpoint save/load, and deterministic seeds.
  - Done when: the model deliberately overfits 16–32 examples and a stopped run resumes from its checkpoint. This tiny overfit test proves the training path works before paying for a full run.

- [ ] **11. Run a pilot, then the final training job**
  - Learn: experiment comparison, hyperparameter tuning, cost tracking, and failure analysis.
  - Build: train on a small subset first, compare against the baseline, inspect failures, then run the justified final configuration on Qwen3-VL-8B-Instruct.
  - Done when: the adapter and processor reload in a fresh process, test metrics beat the baseline on the chosen success criteria, and hardware, duration, cost, seed, and hyperparameters are recorded.

### Phase 3: Serve and document the result

- [ ] **12. Build the FastAPI inference service**
  - Learn: request validation, model startup, dependency injection, error responses, and integration testing with the [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/first-steps/).
  - Build: an endpoint that accepts an image and question, runs the trained adapter, and returns the answer with model metadata.
  - Done when: valid requests return predictions, invalid requests return useful errors, and an integration test passes.

- [ ] **13. Finish the portfolio documentation**
  - Learn: how to explain engineering decisions with evidence.
  - Build: update this README with verified dataset counts, architecture, setup commands, experiment results, limitations, example requests, and lessons learned.
  - Done when: another developer can clone the repository, reproduce evaluation, and understand what worked, what failed, and why.

## Primary references

- [Official Qwen3-VL model collection](https://huggingface.co/collections/Qwen/qwen3-vl)
- [Qwen3-VL Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
- [Official Qwen vision-language fine-tuning framework](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)
- [PyTorch datasets and data loaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Hugging Face PEFT LoRA guide](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
- [Hugging Face bitsandbytes quantization guide](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
