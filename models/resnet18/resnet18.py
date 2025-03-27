import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule


class PneumoniaResNet(LightningModule):
    def __init__(self, num_classes=2, learning_rate=3e-4):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3), 
            nn.Linear(self.model.fc.in_features, num_classes)
        )
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

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

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)  
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, patience=3, factor=0.1, mode="min",min_lr=1e-6),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]