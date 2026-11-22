
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class RetinalDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = Path(image_dir)
        self.labels_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        row = self.labels_df.iloc[index]

        image_name = str(row["id_code"]) + ".png"
        image_path = self.image_dir / image_name
        label = int(row["diagnosis"])

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
