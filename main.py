import json
from functools import partial
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from peft import (
    LoraConfig,
    PeftMixedModel,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import decode_image
from transformers import (
    AutoProcessor,
    BatchFeature,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)

ADAPTER_PATH = "lora_8b_qlora"
CHECKPOINT_PATH = "/workspace/flashcard-generator/checkpoint_path"


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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["question_text", "answer_text"])
    df = df[df["deleted"] == False]
    group_columns = ["storage_path", "page_number"]
    df["chunk"] = df.groupby(group_columns, sort=False).cumcount() // 3
    grouping = df.groupby(
        group_columns + ["chunk"],
        as_index=False,
        sort=False,
    ).agg(
        question_text=("question_text", list),
        answer_text=("answer_text", list),
    )
    return grouping


class CustomDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        df = pd.read_csv("data/sql/questions_rows.csv")
        grouping = clean_data(df)
        empty = pd.read_csv("data/sql/empty_slides.csv")
        empty["question_text"] = empty["question_text"].map(json.loads)
        empty["answer_text"] = empty["answer_text"].map(json.loads)
        grouping = pd.concat([grouping, empty])
        self.dataset = grouping
        self.img_dir = "data/slide_images"
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        img = f"data/images/{row['storage_path']}+{int(row['page_number'])}.png"
        img = decode_image(img)
        label = []
        for i in range(len(row["question_text"])):
            label.append(
                {
                    "question": row["question_text"][i],
                    "answer": row["answer_text"][i],
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
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = prepare_model_for_kbit_training(model)
    checkpoint = Path(CHECKPOINT_PATH) / "adapter_config.json"
    if checkpoint.is_file():
        peft_model = PeftModel.from_pretrained(
            model, CHECKPOINT_PATH, is_trainable=True
        )
    else:
        peft_model = get_peft_model(model, config)
    peft_model.config.text_config.use_cache = False
    peft_model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in peft_model.parameters() if parameter.requires_grad),
        lr=1e-4,
        eps=1e-6,
    )
    num_epochs = 3
    for i in range(num_epochs):
        for step, batch in enumerate(dataloader):
            batch = batch.to("cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = peft_model(**batch)  # forward pass
                loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
            optimizer.step()
            print(f"Step {i}, Loss {loss.item()}")
            print(f"Epoch: {i}, Step: {step}, Loss: {loss.item()}")
            if step % 200 == 0:
                peft_model.save_pretrained(CHECKPOINT_PATH)
        peft_model.save_pretrained(CHECKPOINT_PATH)
    peft_model.save_pretrained(ADAPTER_PATH)
    return peft_model


def inference(
    model: PeftModel | PeftMixedModel | Qwen3VLForConditionalGeneration,
    images: list[Tensor],
    processor: Qwen3VLProcessor,
) -> list[str]:
    model.eval()
    model.config.text_config.use_cache = True
    output_texts = []
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
        print(f"Token count: {response_ids.shape[1]}")
        output_text = processor.batch_decode(
            response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_texts.append(output_text)
    return output_texts


def main():
    load_existing = False
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name, max_pixels=1024 * 1024)
    dataset = CustomDataset()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, quantization_config=quantization_config
    )
    training_set, test_set = random_split(
        dataset,
        [len(dataset) - 50, 50],
        generator=torch.Generator().manual_seed(42),
    )
    dataloader = DataLoader(
        training_set,
        batch_size=2,
        shuffle=True,
        collate_fn=partial(collate_fn, processor=processor),
    )
    if load_existing and Path(ADAPTER_PATH).is_dir():
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        output_text = inference(model, [test_set[0][0]], processor)
        print(f"Expected text: {test_set[0][1]}")
        print(f"Fine tuned text: {output_text[0]}")
    else:
        model = lora_train(model, dataloader)


if __name__ == "__main__":
    main()
