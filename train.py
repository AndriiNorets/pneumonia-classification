import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchvision import transforms

from dataset.datamodule import PneumoniaDataModule
from models.yolo11l.yolo11l import PneumoniaYOLO11L

wandb.login()

# Resnet18, VGG16, CNN, YOLO11L augmentation
train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# CLIP augmentation
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)

data_module = PneumoniaDataModule(
    dataset_link="hf-vision/chest-xray-pneumonia",
    train_transform=train_transform,
    val_transform=val_transform,
    test_transform=val_transform,
    data_dir="./dataset/data",
    batch_size=16,
    num_workers=4,
)

# MODELS---------

# model = PneumoniaResNet(num_classes=2)

# model = PneumoniaVGG16(num_classes=2)

# model = CNNModel(input_channels=3, num_features=32, num_classes=2, learning_rate=3e-4)

model = PneumoniaYOLO11L(num_classes=2)


checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./checkpoints",
    filename="pneumonia-resnet-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
)

early_stop_callback = EarlyStopping(
    monitor="val_loss", patience=100, mode="min", verbose=True, min_delta=0.005
)

wandb_logger = WandbLogger(
    project="pneumonia-classification",
    # name="resnet18",
    # name="vgg16",
    # name="cnn",
    name="yolo11l",
    log_model="all",
    save_dir="./wandb_logs",
)

trainer = pl.Trainer(
    accelerator="auto",
    devices="auto",
    max_epochs=100,
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=wandb_logger,
    deterministic=True,
    log_every_n_steps=10,
    gradient_clip_val=1.0,
    accumulate_grad_batches=4,
)

trainer.fit(model, datamodule=data_module)

trainer.test(model, datamodule=data_module, ckpt_path="best")

wandb.finish()

print("Training complete! Best model saved at:", checkpoint_callback.best_model_path)
