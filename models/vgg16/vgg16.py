import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule


class PneumoniaVGG16(LightningModule):
    def __init__(self, num_classes=2, learning_rate=3e-4):
        super().__init__()
        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # Replace classifier head (original VGG16 has 1000-class output)
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # VGG16 flattened features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.35, 3.85]),  # Class weights
            label_smoothing=0.1,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01, eps=1e-8
        )
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, patience=8, factor=0.1, min_lr=1e-6, verbose=True
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]
