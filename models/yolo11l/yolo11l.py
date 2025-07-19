import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule
from ultralytics import YOLO


# A clear, standard name for the class
class PneumoniaYOLO11L(LightningModule):
    def __init__(self, num_classes=2, learning_rate=3e-4):
        super().__init__()
        self.save_hyperparameters()

        # --- FIX #1: Use the official YOLOv8 model and a temporary loader ---
        # This prevents the 'hijacking' behavior.
        yolo_loader = YOLO("checkpoints/yolo11l-cls.pt")
        self.model = yolo_loader.model
        # --- End of Fix #1 ---

        # --- FIX #2: This code now works because we used the correct model ---
        # The structure of `self.model` is now correct.
        original_head = self.model.model[-1]
        in_features = original_head.linear.in_features
        self.model.model[-1].linear = nn.Linear(in_features, num_classes)
        # --- End of Fix #2 ---

        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.85, 0.69]),
            label_smoothing=0.1,
        )

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
        # --- FIX #3: Unpack the tuple output ---
        y_hat = self(x)[0]
        # --- End of Fix #3 ---
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
        # --- FIX #3: Unpack the tuple output ---
        y_hat = self(x)[0]
        # --- End of Fix #3 ---
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
        # --- FIX #3: Unpack the tuple output ---
        y_hat = self(x)[0]
        # --- End of Fix #3 ---
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
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01
        )
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, patience=8, factor=0.1, min_lr=1e-6, verbose=True
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]
