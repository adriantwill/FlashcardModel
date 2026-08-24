import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def collate_fn(batch: list[tuple[Tensor, str]]):
    inputs_and_outputs = []
    inputs = []
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
    inputs_outputs_tokenize = processor.apply_chat_template(
        inputs_and_outputs,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs_tokensize = processor.apply_chat_template(
        inputs,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
    )
    inputs_outputs_tokenize["labels"] = inputs_outputs_tokenize["input_ids"].clone()
    for i in range(len(inputs_tokensize["input_ids"])):
        for j in range(len(inputs_tokensize["input_ids"][i])):
            if inputs_tokensize["attention_mask"][i][j] == 1:
                inputs_outputs_tokenize["labels"][i][j] = -100
        for j in range(len(inputs_outputs_tokenize["input_ids"][i]) - 1, -1, -1):
            if inputs_outputs_tokenize["attention_mask"][i][j] == 0:
                inputs_outputs_tokenize["labels"][i][j] = -100
    print(inputs_outputs_tokenize["labels"])
    return inputs_outputs_tokenize


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


processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
# )
dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
it = iter(dataloader)
first = next(it)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    dtype=torch.float16,
    # quantization_config=quantization_config,
)
for parameter in model.parameters():
    parameter.requires_grad = False

# Unfreeze only final text transformer block.
for parameter in model.model.language_model.layers[-1].parameters():
    parameter.requires_grad = True

model = model.to("mps")
first = first.to("mps")
model.train()
optimizer = torch.optim.AdamW(
    (parameter for parameter in model.parameters() if parameter.requires_grad),
    lr=1e-5,
    eps=1e-4,
)
for i in range(10):
    optimizer.zero_grad()
    outputs = model(**first)  # forward pass
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    print(f"Step {i}, Loss {loss.item()}")
model.eval()
image, expected = dataset[0]
message = (
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {
                "type": "text",
                "text": "Analyze this educational slide and generate 2-3 flashcard-style questions targeting key facts, definitions, and terms a student would need to memorize for an exam.",
            },
        ],
    },
)
inputs_test = processor.apply_chat_template(
    message,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    padding=True,
    add_generation_prompt=True,
)
inputs_test = inputs_test.to("mps")
generated_ids = model.generate(**inputs_test, max_new_tokens=128)
prompt_length = inputs_test["input_ids"].shape[1]
response_ids = generated_ids[:, prompt_length:]
output_text = processor.batch_decode(
    response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"generated: {output_text}")
print(f"expcted: {expected}")
