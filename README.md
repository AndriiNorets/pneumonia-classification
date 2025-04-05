# Pneumonia Classification from Chest X-Ray Images

A deep learning project for classifying medical X-ray images to detect pneumonia, built with Hugging Face and PyTorch.

## Project Overview

This project aims to develop an accurate deep learning model that can classify chest X-ray images as either normal or showing signs of pneumonia. The solution leverages modern machine learning techniques to assist in medical diagnosis.

## Dataset

This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset loaded via Hugging Face's `datasets` library. The original Kaggle dataset contains 5,863 JPEG images (1,583 Normal, 4,280 Pneumonia) was restructured with a custom 80-10-10 split that maintains class ratios:

| Category  | Samples | Split (80-10-10) |
| --------- | ------- | ---------------- |
| Normal    | 1,583   | 1,266-158-159    |
| Pneumonia | 4,280   | 3,418-427-428    |
| **Total** | 5,863   | 4,684-585-587    |

## Model Architecture

- Base Model: Pre-trained ResNet18
- Custom Head:
  - Additional fully-connected layer (512 units)
  - Batch normalization and ReLU activation
  - Dropout regularization (p=0.2)
  - Binary classification output

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
  
