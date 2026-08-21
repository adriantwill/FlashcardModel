import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image


class CustomDataset(Dataset):
    def __init__(self):
        self.dataset = pd.read_csv("data/sql/question_rows.csv")
        self.img_dir = "data/slide_images"
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        uploads_path = pd.load_csv("TEMP")
        file_name = uploads_path[uploads_path["FILE_NAME"] == row["FILE_NAME"]]
        img = Path(f"data/images/${row['id'] - file_name}")
        img = decode_image(img)
        label = row["question"]
        if self.transform:
            label = self.transform(label)
        return img, label


dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
