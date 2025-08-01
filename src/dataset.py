import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd

class KneeXrayDataset(Dataset):
    def __init__(self, images_dir, csv_file, transform=None):
        self.images_dir = images_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.data.iloc[idx]["image_name"])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))

        angle = self.data.iloc[idx]["angle"]

        if self.transform:
            image = self.transform(image)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        angle = torch.tensor(angle, dtype=torch.float32)

        return image, angle
