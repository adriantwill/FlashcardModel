import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)


def collate_fn(batch: list[tuple[Tensor, str]]):
    messages = []
    model_generated = []
    for tup in batch:
        (img, ques) = tup
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img,
                    },
                    {
                        "type": "text",
                        "text": "Analyze this educational slide and generate 2-3 flashcard-style questions targeting key facts, definitions, and terms a student would need to memorize for an exam.",
                    },
                ],
            },
        ]
        model_generated.append(message.copy())
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
        messages.append(message)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt", padding=True
    )
    outputs = processor.apply_chat_template(
        model_generated,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
    )
    inputs["labels"] = inputs["input_ids"].clone()
    for i in range(len(outputs["input_ids"])):
        for j in range(len(outputs["input_ids"][i])):
            if outputs["attention_mask"][i][j] == 1:
                inputs["labels"][i][j] = -100
        for j in range(len(inputs["input_ids"][i]) - 1, -1, -1):
            if inputs["attention_mask"][i][j] == 0:
                inputs["labels"][i][j] = -100
    return inputs


class CustomDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.dataset = pd.read_csv("data/sql/temp_questions.csv")
        self.img_dir = "data/slide_images"
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        print(row["storage_path"])
        img = f"data/pdf_images/{row['storage_path']}_{int(row['page_number'])}.png"
        img = decode_image(img)
        label = row["question_text"]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)
dataset = CustomDataset()

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    quantization_config=quantization_config,
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
