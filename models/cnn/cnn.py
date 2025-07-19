import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule


class CNNModel(LightningModule):
    def __init__(
        self, input_channels=3, num_features=32, num_classes=2, learning_rate=3e-4
    ):
        super().__init__()
        self.save_hyperparameters()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                input_channels=self.hparams.input_channels,
                out_channels=self.hparams.num_features,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.hparams.num_features),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                input_channels=self.hparams.num_features,
                out_channels=self.hparams.num_features * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.hparams.num_features * 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                input_channels=self.hparams.num_features * 2,
                out_channels=self.hparams.num_features * 2 * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.hparams.num_features * 2 * 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(
                input_channels=self.hparams.num_features * 2 * 2,
                out_channels=self.hparams.num_features * 2 * 2 * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.hparams.num_features * 2 * 2 * 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self._feature_size = self._get_feature_size(
            (self.hparams.input_channels, 224, 224)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._feature_size, num_features * 2 * 2 * 2 * 2),
            nn.BatchNorm1d(self.hparams.num_features * 2 * 2 * 2 * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                self.hparams.num_features * 2 * 2 * 2 * 2, self.hparams.num_classes
            ),
        )

        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.85, 0.69]),
            label_smoothing=0.1,
        )

        self.learning_rate = learning_rate

        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.test_f1 = F1Score(task="binary")

    def _get_feature_size(self, shape):
        dummy_input = torch.zeros(1, *shape)
        x = self.layer1(dummy_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return int(torch.numel(x) / x.shape[0])

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        self.train_acc.update(preds, y)
        self.train_f1.update(preds, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1", self.train_f1, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
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
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        self.test_acc.update(preds, y)
        self.test_f1.update(preds, y)

        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc, on_epoch=True)
        self.log("test_f1", self.test_f1, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01,
            eps=1e-8,
        )
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                patience=8,
                factor=0.1,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]
