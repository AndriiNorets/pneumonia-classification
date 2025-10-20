# Pneumonia Classification from Chest X-Ray Images

A deep learning project for classifying medical X-ray images to detect pneumonia, built with Hugging Face and PyTorch.

## Setting Up the Project Locally

### Prerequisites

- Python 3.11 or higher
- Poetry package manager

### Manual Setup

1. Clone repository in your directory
```bash
git clone https://github.com/AndriiNorets/pneumonia-classification.git
```
2. Go to the repository direction
```bash
cd pneumonia-classification
```
3. Install project dependencies with `Makefile`
```bash
Make install
```

4. Chose `mode` and `model` in `configs/config.yaml`

```yaml
model: # resnet18 / yolo11l / vgg16 / cnn / dinov2
```

```yaml
mode: # debug or train
```

<strong>Debug</strong> - in this mode in will run model for 1 epoch in test mode for check if everything work.
<strong>Train</strong> - in this mode in will run training process with superparametrs in `config/` files

5. Start training process 
```bash
Make train
```

6. In train mode you will be asked for `Wandb` login

```bash
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 
```



## Project Overview

This project aims to develop an accurate deep learning model that can classify chest X-ray images as either normal or showing signs of pneumonia. The solution leverages modern machine learning techniques to assist in medical diagnosis.

## Dataset

This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset loaded via Hugging Face's `datasets` library. The original Kaggle dataset contains 5,863 JPEG images (1,583 Normal, 4,280 Pneumonia) was restructured with a custom 80-10-10 split that maintains class ratios:

| Category  | Samples | Split (80-10-10) |
| --------- | ------- | ---------------- |
| Normal    | 1,583   | 1,266-158-159    |
| Pneumonia | 4,280   | 3,418-427-428    |
| **Total** | 5,863   | 4,684-585-587    |

For class balance:

`configs/model/model_name.yaml`

```yaml
class_weights: [1.85, 0.69]
```

`models/model_name/model_name.py`

```python
weights = torch.tensor(self.hparams.class_weights)
self.criterion = nn.CrossEntropyLoss(weight=weights)
```


## Models

#### ResNet18
| Layer name      | Parametrs |
| --------------- | --------- |
| ResNet18        | 11.4 M    |


#### VGG16
| Layer name      | Parametrs |
| --------------- | --------- |
| VGG16           | 134 M     |

#### YOLO11L
| Layer name      | Parametrs |
| --------------- | --------- |
| YOLO11L         | 12.8 M    |

#### CNN from scratch
| Layer name      | Parametrs |
| --------------- | --------- |
| layer1          | 960       |
| layer2          | 18.6 K    |
| layer3          | 74.1 K    |
| layer4          | 295 K     |
| classifier      | 25.7 M    |

#### DINOV2 + classification head

| Layer name      | Parametrs |
| --------------- | --------- |
| Dinov2Model     | 86.6 M    |
| classifier_head | 396 K     | 

## Training Process

The model was trained using PyTorch Lightning with the following configuration:

- Framework: PyTorch Lightning & Lightning AI Studio
- Monitoring: Weights & Biases (wandb) for real-time metrics tracking
- Training:
  - Batch size: 16 (gradient accumulation: 4 steps)
  - Max epochs: 100 (early stopping patience: 100)
  - Optimizer: AdamW (lr=3e-4, weight decay=0.01)
  - Augmentations: Random crops/flips + normalization
- Validation: Center crops only
- Regularization:
  - Dropout (0.2)
  - Label smoothing (0.1)
  - Class-weighted loss ([1.85, 0.69])
- Checkpoints: Top 3 models saved (val_loss monitored)
- Metrics Tracked: Loss, Accuracy, F1.
  
