import json
from functools import partial
from typing import cast

import pandas as pd
import torch
from peft import LoraConfig, PeftMixedModel, PeftModel, TaskType, get_peft_model
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image
from transformers import (
    AutoProcessor,
    BatchFeature,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)

REDUCE_SIZE = True
MODEL_PATH = "model.pt"
BASE_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
ADAPTER_PATH = "lora"
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def input_message(img: Tensor):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                },
                {
                    "type": "text",
                    "text": """Analyze this educational slide and generate 2-3 flashcard-style questions targeting key facts, definitions, and terms a student would need to memorize for an exam.
Focus on:
- Definitions and terminology
- Key facts, dates, or formulas
- Lists or steps to memorize

Question rules:
- Ask only direct, positive questions about content visible on the slide
- No filler framing: avoid "according to the slide", "in the context of...", "based on...", etc.
- Do not ask about absent content or exclusions: no "NOT", "except", "not mentioned", or "not a symptom/example"
- Avoid questions unrelated to the actual slide content, like names of institutions 

For each question, generate exactly 3 wrong but plausible options based on the slide.
Rules for options:
- Must be incorrect
- Do not paraphrase or restate the correct answer
- Do not use "all/none of the above"
- Keep length similar to correct answer
- Avoid copying long phrases verbatim from the slide
- Skip the slide if it has no testable content

Return JSON array only:
[
  {
    "question": "Question here",
    "answer": "Concise answer without repeating the question",
    "options": ["Wrong but plausible 1", "Wrong but plausible 2", "Wrong but plausible 3"]
  }
]""",
                },
            ],
        }
    ]


def collate_fn(
    batch: list[tuple[Tensor, str]],
    processor: Qwen3VLProcessor,
) -> BatchFeature:
    inputs_and_outputs = []
    inputs = []
    for tup in batch:
        (img, ques) = tup
        message = input_message(img)
        inputs.append(message.copy())
        message.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": ques,
                    },
                ],
            },
        )
        inputs_and_outputs.append(message)
    inputs_outputs_tokenize = cast(
        BatchFeature,
        processor.apply_chat_template(
            inputs_and_outputs,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ),
    )
    inputs_tokensize = cast(
        BatchFeature,
        processor.apply_chat_template(
            inputs,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        ),
    )
    inputs_outputs_tokenize["labels"] = inputs_outputs_tokenize["input_ids"].clone()
    for i in range(len(inputs_tokensize["input_ids"])):
        for j in range(len(inputs_tokensize["input_ids"][i])):
            if inputs_tokensize["attention_mask"][i][j] == 1:
                inputs_outputs_tokenize["labels"][i][j] = -100
        for j in range(len(inputs_outputs_tokenize["input_ids"][i]) - 1, -1, -1):
            if inputs_outputs_tokenize["attention_mask"][i][j] == 0:
                inputs_outputs_tokenize["labels"][i][j] = -100
    return inputs_outputs_tokenize


class CustomDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        df = pd.read_csv("data/sql/questions_rows.csv")
        df = df.dropna(subset=["question_text", "answer_text", "options"])
        grouping = df.groupby(as_index=False, by=["storage_path", "page_number"]).agg(
            question_text=("question_text", list),
            answer_text=("answer_text", list),
            options=("options", list),
        )
        self.dataset = grouping
        self.img_dir = "data/slide_images"
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        img = f"data/pdf_images/{row['storage_path']}_{int(row['page_number'])}.png"
        img = decode_image(img)
        label = []
        for i in range(len(row["question_text"])):
            label.append(
                {
                    "question": row["question_text"][i],
                    "answer": row["answer_text"][i],
                    "options": json.loads(row["options"][i]),
                }
            )
        label = json.dumps(label)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


def lora_train(
    model: Qwen3VLForConditionalGeneration, dataloader: DataLoader
) -> PeftModel | PeftMixedModel:
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, config)
    if REDUCE_SIZE:
        peft_model.gradient_checkpointing_enable()  # reducdes size
        peft_model.config.text_config.use_cache = False  # reduces size
    peft_model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in peft_model.parameters() if parameter.requires_grad),
        lr=1e-4,
        eps=1e-4,
    )
    num_epochs = 2
    for i in range(num_epochs):
        for step, batch in enumerate(dataloader):
            batch = batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            outputs = peft_model(**batch)  # forward pass
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
            optimizer.step()
            print(f"Step {i}, Loss {loss.item()}")
            print(f"Epoch: {i}, Step: {step}, Loss: {loss.item()}")
            if REDUCE_SIZE:
                del batch, outputs, loss
                torch.mps.empty_cache()
    peft_model.save_pretrained(ADAPTER_PATH)
    return peft_model


def partial_train_model(
    model: Qwen3VLForConditionalGeneration, dataloader: DataLoader
) -> Qwen3VLForConditionalGeneration:
    print("Parital full training")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.model.language_model.layers[-1].parameters():
        parameter.requires_grad = True
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-5,
        eps=1e-4,
    )
    for i in range(10):  # epoch
        it = iter(dataloader)
        for j in range(8):  # trainign step
            first = next(it)
            first = first.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**first)  # forward pass
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            print(f"Epoch {i}, Step {j}, Loss {loss.item()}")
            if REDUCE_SIZE:
                del first, outputs, loss
                torch.mps.empty_cache()
    model.save_pretrained(MODEL_PATH)
    return model


def inference(
    model: PeftModel | PeftMixedModel | Qwen3VLForConditionalGeneration,
    image: Tensor,
    processor: Qwen3VLProcessor,
) -> str:
    model.eval()
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
    inputs_test = inputs_test.to(DEVICE)
    generated_ids = model.generate(**inputs_test, max_new_tokens=512)
    prompt_length = inputs_test["input_ids"].shape[1]
    response_ids = generated_ids[:, prompt_length:]
    print(f"Token count: {response_ids.shape[1]}")
    output_text = processor.batch_decode(
        response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text


def main():
    load_existing = True
    processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
    dataset = CustomDataset()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_NAME, dtype=torch.bfloat16
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=partial(collate_fn, processor=processor),
    )
    if load_existing:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    else:
        model = lora_train(model, dataloader)
    model = model.to(DEVICE)
    output_text = inference(model, dataset[0][0], processor)
    print(f"Expected text: {dataset[0][1]}")
    print(f"Fine tuned text: {output_text}")


if __name__ == "__main__":
    main()
