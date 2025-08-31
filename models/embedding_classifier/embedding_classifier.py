import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule
from transformers import AutoModel, AutoProcessor


class EmbeddingClassifier(LightningModule):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        num_classes: int = 2,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        print(f"Loading pre-trained backbone: {self.hparams.model_name}")

        self.backbone = AutoModel.from_pretrained(self.hparams.model_name)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.embedding_size = self.backbone.config.hidden_size

        self.classifier_head = nn.Sequential(
            nn.LayerNorm(self.embedding_size),
            nn.Linear(self.embedding_size, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, self.hparams.num_classes), 
        )

        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.85, 0.69]))

        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.test_f1 = F1Score(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)

        embedding = outputs.last_hidden_state[:, 0, :]

        return self.classifier_head(embedding)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        self.train_acc.update(preds, y)
        self.train_f1.update(preds, y)

        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": self.train_acc,
                "train_f1": self.train_f1,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)

        self.log_dict(
            {"val_loss": loss, "val_acc": self.val_acc, "val_f1": self.val_f1},
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        self.test_acc.update(preds, y)
        self.test_f1.update(preds, y)

        self.log_dict(
            {"test_loss": loss, "test_acc": self.test_acc, "test_f1": self.test_f1},
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.classifier_head.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.05,
        )

        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.7, verbose=True
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]
