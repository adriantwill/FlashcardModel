import base64
import binascii
import json
import os
from io import BytesIO
from pathlib import Path
from typing import cast

import runpod
import torch
from peft import PeftModel
from PIL import Image, UnidentifiedImageError
from transformers import (
    AutoProcessor,
    BatchFeature,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
ADAPTER_PATH = Path(__file__).resolve().parent / "adapter"
RUNPOD_MODEL_CACHE = Path("/runpod-volume/huggingface-cache/hub")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


def input_message(image: Image.Image):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": """Analyze this educational slide and generate 1-3 flashcard-style questions targeting key facts, definitions, and terms a student would need to memorize for an exam.
Focus on:
- Definitions and terminology
- Key facts, dates, or formulas
- Lists or steps to memorize

Question rules:
- Ask only direct, positive questions about content visible on the slide
- No filler framing: avoid "according to the slide", "in the context of...", "based on...", etc.
- Do not ask about absent content or exclusions: no "NOT", "except", "not mentioned", or "not a symptom/example"
- Avoid questions unrelated to the actual slide content, like names of institutions
- If no testable content, return empty array with no questions

Return JSON array only:
[
  {
    "question": "Question here",
    "answer": "Concise answer without repeating the question"
  }
]""",
                },
            ],
        }
    ]


def inference(
    model: PeftModel | Qwen3VLForConditionalGeneration,
    images: list[Image.Image],
    processor: Qwen3VLProcessor,
) -> list[str]:
    model.eval()
    model.config.text_config.use_cache = True
    output_texts = []
    with torch.inference_mode():
        for image in images:
            inputs_test = cast(
                BatchFeature,
                processor.apply_chat_template(
                    input_message(image),
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                    add_generation_prompt=True,
                ),
            )
            inputs_test = inputs_test.to("cuda")
            generated_ids = model.generate(**inputs_test, max_new_tokens=512)
            prompt_length = inputs_test["input_ids"].shape[1]
            response_ids = generated_ids[:, prompt_length:]
            output_text = processor.batch_decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            output_texts.append(output_text)
    return output_texts


def model_source() -> str:
    model_root = RUNPOD_MODEL_CACHE / f"models--{MODEL_NAME.replace('/', '--')}"
    main_ref = model_root / "refs" / "main"
    if main_ref.is_file():
        snapshot = model_root / "snapshots" / main_ref.read_text().strip()
        if snapshot.is_dir():
            return str(snapshot)
    return MODEL_NAME


source = model_source()
processor = cast(
    Qwen3VLProcessor,
    AutoProcessor.from_pretrained(source, max_pixels=1024 * 1024),
)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    source,
    quantization_config=quantization_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)


def handler(job):
    image_base64 = job.get("input", {}).get("image_base64")
    if not image_base64:
        return {"error": 'Missing input field "image_base64".'}

    if image_base64.startswith("data:"):
        image_base64 = image_base64.split(",", 1)[-1]

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            return {"error": "Image exceeds 10 MB limit."}
        image = Image.open(BytesIO(image_bytes))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            return {"error": "Image pixel count exceeds limit."}
        image = image.convert("RGB")
    except (binascii.Error, UnidentifiedImageError, OSError):
        return {"error": "image_base64 is not a valid image."}

    output = inference(model, [image], processor)[0]
    try:
        return {"flashcards": json.loads(output)}
    except json.JSONDecodeError:
        return {"error": "Model returned invalid JSON.", "raw_output": output}


runpod.serverless.start({"handler": handler})
