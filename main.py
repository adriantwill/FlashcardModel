import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image


class CustomDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.dataset = pd.read_csv("data/sql/questions_rows.csv")
        self.img_dir = "data/slide_images"
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        img = f"data/images/{row['id']} - {row['page_number']}"
        img = decode_image(img)
        label = row["question"]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
train_features, train_labels = next(iter(dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
