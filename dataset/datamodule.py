from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pytorch_lightning import LightningDataModule
import torchvision
import kaggle
import os


class PneumoniaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.dataset = ImageFolder(root=self.data_dir, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class PneumoniaDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_link,
        train_transform,
        val_transform,
        test_transform,
        data_dir=".",
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.dataset_link = dataset_link
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        if os.path.exists(self.data_dir) and os.listdir(self.data_dir):
            print(f"Dataset already exists at {self.data_dir}. Skipping download.")
        else:
            kaggle.api.authenticate()
            dataset_path = kaggle.api.dataset_download_files(
                self.dataset_link, path=self.data_dir, unzip=True
            )

        print(f"Dataset extracted to {self.data_dir}")

    def setup(self, stage=None):
        self.train_dataset = PneumoniaDataset(
            data_dir=f"{self.data_dir}/train", transform=self.train_transform
        )

        self.val_dataset = PneumoniaDataset(
            data_dir=f"{self.data_dir}/val", transform=self.test_transform
        )

        self.test_dataset = PneumoniaDataset(
            data_dir=f"{self.data_dir}/test", transform=self.test_transform
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
            batch_size=int(self.batch_size / 2),
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
