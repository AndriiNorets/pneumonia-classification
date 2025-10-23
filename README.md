# Pneumonia Classification from Chest X-Ray Images

A deep learning project for classifying medical X-ray images to classify lungs for pneumonia, built with Hugging Face and PyTorch.

Web application with models usage functionality on Huggingface: [Link to the application](<https://huggingface.co/spaces/AndriiNorets/pneumonia-classification>)

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

3. Activate your environment
```bash
poetry shell
```

4. Install Make

    <strong>Windows</strong>

    - Open a PowerShell terminal (version 5.1 or later) and from the PS C:\> prompt, run: 
    ```bash
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
    ```

    - Install Make
    ```bash
    scoop install make
    ```

    <strong>Linux</strong>
    - Install Make
    ```bash
    sudo apt update
    ```

    ```bash
    sudo apt install make
    ```


5. Install project dependencies with `Makefile`
```bash
Make install
```

6. Chose `mode` and `model` in `configs/config.yaml`

```yaml
model: # resnet18 / yolo11l / vgg16 / cnn / dinov2
```

```yaml
mode: # debug or train
```

<strong>Debug</strong> - in this mode in will run model for 1 epoch in test mode for check if everything work.
<strong>Train</strong> - in this mode in will run training process with superparametrs in `config/` files

7. Start training process 
```bash
Make train
```

8. In train mode you will be asked for `Wandb` login

```bash
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 
```



## Project Overview

This project aims to develop an accurate comparison of transfer learning and training from scratch deep learning methodologies in pneumonia classification based on X-ray images. 

## Project Structure

```
├── .github/                 # GitHub settings (e.g., workflows)
├── configs/                 # Main directory for Hydra configurations
│   ├── config.yaml          # Main config file that orchestrates experiments
│   ├── data/
│   │   └── pneumonia.yaml   # Configuration for the DataModule
│   ├── model/
│   │   ├── cnn.yaml
│   │   ├── dinov2.yaml
│   │   ├── resnet18.yaml
│   │   ├── vgg16.yaml
│   │   └── yolo11l.yaml
│   └── trainer/
│       └── default.yaml     # Configuration for the Trainer
│
├── dataset/                 # Code for download data from HuggingFace
│   ├── __init__.py
│   └── datamodule.py
│
├── model_weights/           # Locally stored YOLO11L model weights
│   └── yolo11l-cls.pt
│
├── models/                  
│   ├── cnn/
│   │   ├── __init__.py
│   │   └── cnn.py
│   ├── embedding_classifier/
│   │   ├── __init__.py
│   │   └── embedding_classifier.py
│   ├── resnet18/
│   │   ├── __init__.py
│   │   └── resnet18.py
│   ├── vgg16/
│   │   ├── __init__.py
│   │   └── vgg16.py
│   └── yolo11l/
│       ├── __init__.py
│       └── yolo11l.py
│
├── tests/                  
│   ├── __init__.py
│   └── datamodule_test.py   # Test for Datamodule  
│
├── .gitignore               # Files ignored by Git
├── Makefile                 # Commands to manage the project
├── README.md                # Project description
├── check_poetry_version.sh  # Poetry version checker script
├── model_embeddings_comprison.py 
├── poetry.lock              # Locked dependency versions
├── pyproject.toml           # Poetry configuration file
└── train.py                 # Main script for running the training process
```

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
18-layer convolutional neural network 
| Layer name      | Parametrs |
| --------------- | --------- |
| ResNet18        | 11.4 M    |


#### VGG16
6-layer convolutional neural network 
| Layer name      | Parametrs |
| --------------- | --------- |
| VGG16           | 134 M     |

#### YOLO11L-cls
pretrained convolutional neural network model from `ultralytics` for image classification
| Layer name      | Parametrs |
| --------------- | --------- |
| YOLO11L         | 12.8 M    |

#### CNN from scratch
5-layer convolutional neural network 
| Layer name      | Parametrs |
| --------------- | --------- |
| layer1          | 960       |
| layer2          | 18.6 K    |
| layer3          | 74.1 K    |
| layer4          | 295 K     |
| classifier      | 25.7 M    |

#### DINOV2 + classification head
self-supervised Vision Transformer model with lassification head linear layer
| Layer name      | Parametrs |
| --------------- | --------- |
| Dinov2Model     | 86.6 M    |
| classifier_head | 396 K     | 

  
