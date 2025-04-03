import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchvision import transforms

from dataset.datamodule import PneumoniaDataModule
from models.resnet18.resnet18 import PneumoniaResNet
from models.vgg16.vgg16 import PneumoniaVGG16


wandb.login()

# train_transform = transforms.Compose(
#     [
#         transforms.Resize(256),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(p=0.2),
#         transforms.RandomRotation(10),  # Reduced from 15
#         transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Reduced intensity
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#         transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Added
#     ]
# )
train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),  # Most impactful augmentation
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

data_module = PneumoniaDataModule(
    dataset_link="hf-vision/chest-xray-pneumonia",
    train_transform=train_transform,
    val_transform=val_transform,
    test_transform=val_transform,
    data_dir="./dataset/data",
    batch_size=16,
    num_workers=4,
)

model = PneumoniaResNet(num_classes=2)
# model = PneumoniaVGG16(num_classes=2)

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
    name="resnet18",
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
