import pytest
import torch
from unittest.mock import patch, MagicMock
from dataset.datamodule import PneumoniaDataModule
import torchvision.transforms.v2 as transforms
import structlog
import os


@pytest.fixture
def pneumonia_data():
    dataset_link = "paultimothymooney/chest-xray-pneumonia"
    data_dir = "./dataset/data/chest_xray"
    batch_size = 16
    num_workers = 4

    train_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return PneumoniaDataModule(
        dataset_link=dataset_link,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        train_transform=train_transforms,
        val_transform=val_transforms,
        test_transform=test_transforms,
    )


def test_prepare_data(pneumonia_data):
    print("Preparing data...")
    pneumonia_data.prepare_data()

    expected_dirs = ["train", "val", "test"]
    for dir_name in expected_dirs:
        dir_path = f"{pneumonia_data.data_dir}/{dir_name}"
        assert os.path.exists(dir_path), f"Directory {dir_path} does not exist"
        assert os.path.isdir(dir_path), f"{dir_path} is not a directory"

    print("Directory check completed successfully.")


def test_setup_datasets(pneumonia_data):
    print("Setting up datasets...")
    pneumonia_data.prepare_data()
    pneumonia_data.setup()

    assert pneumonia_data.train_dataset is not None, "Train dataset not created"
    assert pneumonia_data.val_dataset is not None, "Validation dataset not created"
    assert pneumonia_data.test_dataset is not None, "Test dataset not created"

    print("\nDataset sizes:")
    print(f"Train: {len(pneumonia_data.train_dataset)}")
    print(f"Val: {len(pneumonia_data.val_dataset)}")
    print(f"Test: {len(pneumonia_data.test_dataset)}")


def test_train_dataloader(pneumonia_data):
    print("Testing train_dataloader...")
    pneumonia_data.prepare_data()
    pneumonia_data.setup()
    train_loader = pneumonia_data.train_dataloader()

    logger = structlog.get_logger()

    batch = next(iter(train_loader), None)
    assert batch is not None, "Train dataloader batch is empty"
    assert len(batch) == 2, "Train dataloader batch should contain images and labels"
    assert batch[0].shape == (pneumonia_data.batch_size, 3, 224, 224), (
        "Incorrect image shape in train batch"
    )
    logger.info("Sampled Train Batch", batch=batch)


def test_val_dataloader(pneumonia_data):
    print("Testing val_dataloader...")
    pneumonia_data.prepare_data()
    pneumonia_data.setup()
    val_loader = pneumonia_data.val_dataloader()

    logger = structlog.get_logger()

    batch = next(iter(val_loader), None)
    assert batch is not None, "Validation dataloader batch is empty"
    assert len(batch) == 2, (
        "Validation dataloader batch should contain images and labels"
    )
    assert batch[0].shape == (int(pneumonia_data.batch_size), 3, 224, 224), (
        "Incorrect image shape in validation batch"
    )
    logger.info("Sampled Val Batch", batch=batch)


def test_test_dataloader(pneumonia_data):
    print("Testing test_dataloader...")
    pneumonia_data.prepare_data()
    pneumonia_data.setup()
    test_loader = pneumonia_data.test_dataloader()

    logger = structlog.get_logger()

    batch = next(iter(test_loader), None)
    assert batch is not None, "Test dataloader batch is empty"
    assert len(batch) == 2, "Test dataloader batch should contain images and labels"
    assert batch[0].shape == (pneumonia_data.batch_size, 3, 224, 224), (
        "Incorrect image shape in test batch"
    )
    logger.info("Sampled Test Batch", batch=batch)
