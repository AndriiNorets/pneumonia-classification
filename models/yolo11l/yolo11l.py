import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule
from ultralytics import YOLO
from typing import List


class PneumoniaYOLO11L(LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        learning_rate: float,
        min_learning_rate: float,
        weight_decay: float,
        class_weights: List[float],
    ):
        super().__init__()
        self.save_hyperparameters()

        yolo_loader = YOLO(f"model_weights/{self.hparams.model_name}.pt")
        self.model = yolo_loader.model

        original_head = self.model.model[-1]
        in_features = original_head.linear.in_features
        self.model.model[-1].linear = nn.Linear(in_features, self.hparams.num_classes)

        weights = torch.tensor(self.hparams.class_weights)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.test_f1 = F1Score(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)[0]

        loss = self.criterion(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        self.train_acc.update(preds, y)
        self.train_f1.update(preds, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)[0]

        loss = self.criterion(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)[0]

        loss = self.criterion(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        self.test_acc.update(preds, y)
        self.test_f1.update(preds, y)

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_acc, on_epoch=True)
        self.log("test_f1", self.test_f1, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                patience=8,
                factor=0.7,
                min_lr=self.hparams.min_learning_rate,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]
