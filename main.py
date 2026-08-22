import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

def collate_fn(batch: list[Tensor, str]):



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

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    quantization_config=quantization_config,
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
