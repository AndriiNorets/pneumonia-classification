import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms
import wandb
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf

from dataset.datamodule import PneumoniaDataModule
from models.resnet18.resnet18 import PneumoniaResNet
from models.vgg16.vgg16 import PneumoniaVGG16
from models.yolo11l.yolo11l import PneumoniaYOLO11L
from models.cnn.cnn import CNNModel
from models.embedding_classifier.embedding_classifier import EmbeddingClassifier

initialize(config_path="configs", version_base=None)
cfg = compose(config_name="config")


# Resnet18, VGG16, CNN, YOLO11L augmentations
# train_transform = transforms.Compose(
#     [
#         transforms.Resize(256),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#     ]
# )

# val_transform = transforms.Compose(
#     [
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ]
# )

# DINOV2 augmentations
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

data_module = PneumoniaDataModule(
    **cfg.data,
    train_transform=train_transform,
    val_transform=val_transform,
    test_transform=val_transform,
)


# MODELS---------

# model = PneumoniaResNet(num_classes=2, learning_rate=3e-4)

# model = PneumoniaVGG16(num_classes=2, learning_rate=3e-4)

# model = CNNModel(input_channels=3, num_features=32, num_classes=2, learning_rate=3e-4)

# model = PneumoniaYOLO11L(num_classes=2, learning_rate=3e-4)

model = hydra.utils.instantiate(cfg.model)

def debug():    
    trainer = pl.Trainer(**cfg.trainer.debug_params)
    
    print("Run test()...")
    trainer.test(model, datamodule=data_module)
    
    print("Debug run complete.")

def train():
    wandb.login()
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename=f"{cfg.model.model_name.replace('/', '_')}-best-{{epoch:02d}}-{{val_f1:.4f}}",
        save_top_k=3,
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=20, mode="min", verbose=True, min_delta=0.005
    )

    wandb_logger = WandbLogger(
        project=cfg.project_name,
        name=cfg.run_name,
        log_model="all",
        save_dir="./wandb_logs",
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module, ckpt_path="best")

    wandb.finish()

    print("Training run complete! Best model saved at:", checkpoint_callback.best_model_path)


if __name__ == "__main__":
    if cfg.mode == "debug":
        print("\n--- Running in DEBUG mode")
        debug()
    elif cfg.mode == "train":
        print("\n--- Running in TRAIN mode")
        train()
 