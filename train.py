import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchvision import transforms

from dataset.datamodule import PneumoniaDataModule
from models.resnet18.resnet18 import PneumoniaResNet


wandb.login()

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

data_module = PneumoniaDataModule(
    dataset_link="paultimothymooney/chest-xray-pneumonia",
    train_transform=train_transform,
    val_transform=val_transform,
    test_transform=val_transform,
    data_dir="./dataset/data/chest_xray",
    batch_size=16,
    num_workers=0,
)

model = PneumoniaResNet(num_classes=2, learning_rate=1e-4)

# Define callbacks
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./checkpoints",
    filename="pneumonia-resnet-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
)

early_stop_callback = EarlyStopping(
    monitor="val_loss", patience=5, mode="min", verbose=True
)

wandb_logger = WandbLogger(
    project="pneumonia-classification",
    name="resnet18",
    log_model="all",
    save_dir="./wandb_logs",
)

trainer = pl.Trainer(
    accelerator="auto",
    devices="auto",
    max_epochs=20,
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=wandb_logger,
    deterministic=True,
    log_every_n_steps=10,
)

trainer.fit(model, datamodule=data_module)

trainer.test(model, datamodule=data_module)

wandb.finish()

print("Training complete! Best model saved at:", checkpoint_callback.best_model_path)
