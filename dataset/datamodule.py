import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from pytorch_lightning import LightningDataModule
import torchvision
from datasets import load_dataset
from PIL import Image
import numpy as np
from datasets import load_dataset
import os
import io


class PneumoniaDataSet(Dataset):
    def __init__(self, hf_data, transform=None):
        self.data = hf_data
        self.transform = (
            transform if transform is not None else self._default_transform()
        )

    def _default_transform(self):
        return v2.Compose(
            [
                v2.Resize((224, 224)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        try:
            image = sample["image"]
            if isinstance(image, dict) and "bytes" in image:
                image = Image.open(io.BytesIO(image["bytes"]))
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            image = image.convert("RGB")
            image = self.transform(image)
            label = torch.tensor(sample["label"], dtype=torch.long)

        except Exception as e:
            raise ValueError(f"Error processing sample {idx}: {e}")

        return image, label


class PneumoniaDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_link,
        train_transform,
        val_transform,
        test_transform,
        data_files="./data",
        data_dir=".",
        batch_size=16,
        num_workers=4,
    ):
        super().__init__()
        self.dataset_link = dataset_link
        self.data_files = data_files
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self.data_dict = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        self.data_dict = load_dataset(
            self.dataset_link,
            cache_dir=self.data_dir,
        )

        print(f"Dataset extracted to {self.data_dir}")

    def setup(self, stage=None):
        self.train_dataset = PneumoniaDataSet(
            self.data_dict["train"], transform=self.train_transform
        )

        self.val_dataset = PneumoniaDataSet(
            self.data_dict["validation"], transform=self.val_transform
        )

        self.test_dataset = PneumoniaDataSet(
            self.data_dict["test"], transform=self.test_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.batch_size),
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
