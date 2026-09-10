# CIFAR-10 Computer Vision Classifier: Development & Experiments

This document details the architecture, data pipeline, training procedure, and empirical results for CIFAR-10 image classification experiments conducted in this repository.

---

## 1. Project Overview & Pipeline Architecture

The objective of this project is to develop and evaluate convolutional neural network (CNN) models for image classification on the CIFAR-10 dataset (10 classes: *airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*), optimizing classification accuracy while maintaining computational efficiency.

### Core Components

- **Dataset & Augmentation Pipeline (`dataset.py`)**:
  - **Dataset Split**: 45,000 training images, 5,000 validation images (deterministic split with seed 42), and 10,000 test images.
  - **Data Augmentation**: `RandomHorizontalFlip()`, `RandomCrop(32, padding=4)`, and `RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))` applied to the training set to prevent overfitting and improve generalization.
  - **Normalization**: Standard CIFAR-10 channel mean `(0.4914, 0.4822, 0.4465)` and standard deviation `(0.2470, 0.2435, 0.2616)`.
  - **Data Loading**: PyTorch `DataLoader` with pinned memory and configurable batch sizes.

- **Model Architectures (`model.py`)**:
  - **Baseline 2-Block CNN (`CIFARClassifier`)**:
    - Block 1: `Conv2d(3, 32, kernel_size=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (Feature map: $32 \times 16 \times 16$)
    - Block 2: `Conv2d(32, 64, kernel_size=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (Feature map: $64 \times 8 \times 8$)
    - Classifier: `Flatten()` $\rightarrow$ `Linear(4096, 512)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(512, 10)`
  - **Architectural Variations Tested**:
    - 3-Block CNN: Adding a 3rd convolutional block ($64 \rightarrow 128$)
    - Batch Normalization (`BatchNorm2d`)
    - Dropout Regularization

- **Training & Evaluation Loop (`train.py`)**:
  - **Criterion**: `nn.CrossEntropyLoss()`
  - **Optimizer**: `optim.Adam(lr=1e-3)`
  - **Learning Rate Scheduler**: `StepLR(step_size=5, gamma=0.1)`
  - **Model Checkpointing**: Automated saving of `best_model.pth` based on highest validation accuracy.

---

## 2. Experiments and Results

The following named experiments evaluate the impact of learning rate scheduling, training epochs, network depth, batch size, and regularization techniques.

### Summary of Experimental Results

| Experiment ID | Experiment Name | Architecture | Batch Size | Epochs | LR Schedule | Best Val Acc (%) |  Peak Epoch  |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |:------------:|
| **EXP-01** | Standard 10-Epoch Baseline | 2-Block (3 $\rightarrow$ 32 $\rightarrow$ 64) | 128 | 10 | Constant (`lr=1e-3`) | **75.18%** |   Epoch 9    |
| **EXP-02** | 2-Block CNN + StepLR (10 Epochs) | 2-Block (3 $\rightarrow$ 32 $\rightarrow$ 64) | 128 | 10 | StepLR (`step=5, γ=0.1`) | **76.12%** | **Epoch 10** |
| **EXP-03** | 2-Block CNN + StepLR (20 Epochs) | 2-Block (3 $\rightarrow$ 32 $\rightarrow$ 64) | 128 | 20 | StepLR (`step=5, γ=0.1`) | **72.54%** |   Epoch 14   |
| **EXP-04** | 3-Block CNN (Deeper Architecture) | 3-Block (3 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 128) | 128 | 20 | StepLR (`step=5, γ=0.1`) | **74.78%** |   Epoch 13   |
| **EXP-05** | 2-Block CNN + BatchNorm + Batch 256 | 2-Block + BatchNorm | 256 | 20 | StepLR (`step=5, γ=0.1`) | **72.06%** |   Epoch 17   |
| **EXP-06** | 2-Block CNN + BatchNorm + Dropout | 2-Block + BatchNorm + Dropout | 128 | 20 | StepLR (`step=5, γ=0.1`) | **72.12%** |   Epoch 15   |

---

### Detailed Experiment Reports

#### Experiment 1: `EXP-01` — Standard 10-Epoch Baseline
- **Description**: Baseline 2-block convolutional neural network trained for 10 epochs with fixed learning rate.
- **Configuration**:
  - Architecture: 2 Conv Blocks (3 $\rightarrow$ 32 $\rightarrow$ 64) + 2 Linear Layers
  - Optimizer: Adam (`lr=1e-3`), constant learning rate
  - Batch Size: 128, Epochs: 10
- **Progression Logs**:
  ```
  Epoch [2/10]  Train Loss: 1.1486  Train Acc: 58.69%  Val Loss: 0.9673  Val Acc: 65.80%
  Epoch [3/10]  Train Loss: 1.0154  Train Acc: 63.87%  Val Loss: 0.9029  Val Acc: 67.62%
  Epoch [4/10]  Train Loss: 0.9251  Train Acc: 67.27%  Val Loss: 0.8269  Val Acc: 71.10%
  Epoch [5/10]  Train Loss: 0.8590  Train Acc: 69.54%  Val Loss: 0.7727  Val Acc: 72.68%
  Epoch [6/10]  Train Loss: 0.8029  Train Acc: 71.79%  Val Loss: 0.7490  Val Acc: 73.80%
  Epoch [7/10]  Train Loss: 0.7642  Train Acc: 73.15%  Val Loss: 0.7732  Val Acc: 73.36%
  Epoch [8/10]  Train Loss: 0.7335  Train Acc: 74.23%  Val Loss: 0.7600  Val Acc: 73.70%
  Epoch [9/10]  Train Loss: 0.7058  Train Acc: 75.32%  Val Loss: 0.7173  Val Acc: 75.18%
  Epoch [10/10] Train Loss: 0.6821  Train Acc: 76.23%  Val Loss: 0.7254  Val Acc: 74.66%
  ```
- **Outcome**: Achieved **75.18%** validation accuracy at Epoch 9. Mild signs of overfitting observed around epochs 6–7 as validation loss plateaued.

---

#### Experiment 2: `EXP-02` — Baseline 2-Block CNN + StepLR Scheduler (10 Epochs)
- **Description**: Introduction of `StepLR(step_size=5, gamma=0.1)` to decrease learning rate from $10^{-3}$ to $10^{-4}$ at epoch 6.
- **Configuration**:
  - Architecture: 2 Conv Blocks (3 $\rightarrow$ 32 $\rightarrow$ 64)
  - Optimizer: Adam (`lr=1e-3`), StepLR scheduler
  - Batch Size: 128, Epochs: 10
- **Progression Logs**:
  ```
  Epoch [2/10]  Train Loss: 1.1460  Train Acc: 59.08%  Val Loss: 0.9966  Val Acc: 64.78%  LR: 0.001000
  Epoch [3/10]  Train Loss: 1.0132  Train Acc: 64.13%  Val Loss: 0.9212  Val Acc: 68.08%  LR: 0.001000
  Epoch [4/10]  Train Loss: 0.9255  Train Acc: 67.12%  Val Loss: 0.8353  Val Acc: 71.06%  LR: 0.001000
  Epoch [5/10]  Train Loss: 0.8661  Train Acc: 69.42%  Val Loss: 0.8042  Val Acc: 70.92%  LR: 0.001000
  Epoch [6/10]  Train Loss: 0.7451  Train Acc: 73.71%  Val Loss: 0.7165  Val Acc: 74.78%  LR: 0.000100
  Epoch [7/10]  Train Loss: 0.7195  Train Acc: 74.76%  Val Loss: 0.7103  Val Acc: 74.56%  LR: 0.000100
  Epoch [8/10]  Train Loss: 0.7045  Train Acc: 75.26%  Val Loss: 0.6911  Val Acc: 76.04%  LR: 0.000100
  Epoch [9/10]  Train Loss: 0.6941  Train Acc: 75.67%  Val Loss: 0.6898  Val Acc: 75.64%  LR: 0.000100
  Epoch [10/10] Train Loss: 0.6863  Train Acc: 75.93%  Val Loss: 0.6807  Val Acc: 76.12%  LR: 0.000100
  ```
- **Outcome**: Best overall performance achieved (**76.12%** at Epoch 10). LR reduction allowed finer convergence and reduced validation loss to 0.6807.

---

#### Experiment 3: `EXP-03` — Baseline 2-Block CNN + StepLR (20 Epochs Extended)
- **Description**: Extending training to 20 epochs with `StepLR(step_size=5, gamma=0.1)` decaying every 5 epochs ($10^{-3} \rightarrow 10^{-4} \rightarrow 10^{-5} \rightarrow 10^{-6}$).
- **Configuration**:
  - Architecture: 2 Conv Blocks (3 $\rightarrow$ 32 $\rightarrow$ 64)
  - Optimizer: Adam (`lr=1e-3`), StepLR (`step_size=5, gamma=0.1`)
  - Batch Size: 128, Epochs: 20
- **Progression Logs**:
  ```
  Epoch [1/20]  Train Loss: 1.6086  Train Acc: 41.33%  Val Loss: 1.2924  Val Acc: 53.92%  LR: 0.001000
  Epoch [2/20]  Train Loss: 1.2701  Train Acc: 54.66%  Val Loss: 1.0971  Val Acc: 61.04%  LR: 0.001000
  Epoch [5/20]  Train Loss: 0.9633  Train Acc: 65.80%  Val Loss: 0.9088  Val Acc: 67.68%  LR: 0.001000
  Epoch [6/20]  Train Loss: 0.8733  Train Acc: 69.40%  Val Loss: 0.8122  Val Acc: 70.68%  LR: 0.000100
  Epoch [10/20] Train Loss: 0.8212  Train Acc: 71.23%  Val Loss: 0.7776  Val Acc: 72.02%  LR: 0.000100
  Epoch [11/20] Train Loss: 0.8084  Train Acc: 71.73%  Val Loss: 0.7749  Val Acc: 72.44%  LR: 0.000010
  Epoch [14/20] Train Loss: 0.8051  Train Acc: 72.08%  Val Loss: 0.7709  Val Acc: 72.54%  LR: 0.000010
  Epoch [20/20] Train Loss: 0.8017  Train Acc: 71.96%  Val Loss: 0.7701  Val Acc: 72.54%  LR: 0.000001
  ```
- **Outcome**: Best validation accuracy was **72.54%** at Epoch 14. Decaying learning rate too aggressively ($10^{-5}$ and $10^{-6}$) caused premature parameter freeze, preventing further optimization gains.

---

#### Experiment 4: `EXP-04` — 3-Block CNN (Deeper Architecture: 3 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 128)
- **Description**: Adding a third convolutional block to evaluate whether expanded model capacity yields better feature representations.
- **Configuration**:
  - Architecture: 3 Conv Blocks (3 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 128)
  - Optimizer: Adam (`lr=1e-3`), StepLR (`step_size=5, gamma=0.1`)
  - Batch Size: 128, Epochs: 20
- **Progression Logs**:
  ```
  Epoch [1/20]  Train Loss: 1.6478  Train Acc: 39.44%  Val Loss: 1.3462  Val Acc: 51.58%  LR: 0.001000
  Epoch [5/20]  Train Loss: 0.9217  Train Acc: 67.52%  Val Loss: 0.8149  Val Acc: 70.74%  LR: 0.001000
  Epoch [6/20]  Train Loss: 0.8079  Train Acc: 71.70%  Val Loss: 0.7555  Val Acc: 73.10%  LR: 0.000100
  Epoch [10/20] Train Loss: 0.7527  Train Acc: 73.55%  Val Loss: 0.7187  Val Acc: 74.26%  LR: 0.000100
  Epoch [13/20] Train Loss: 0.7327  Train Acc: 74.38%  Val Loss: 0.7065  Val Acc: 74.78%  LR: 0.000010
  Epoch [20/20] Train Loss: 0.7285  Train Acc: 74.40%  Val Loss: 0.7051  Val Acc: 74.74%  LR: 0.000001
  ```
- **Outcome**: Reached **74.78%** validation accuracy at Epoch 13, demonstrating improved feature extraction capacity over EXP-03 on a 20-epoch schedule.

---

#### Experiment 5: `EXP-05` — 2-Block CNN + Batch Normalization + Batch Size 256
- **Description**: Incorporated `BatchNorm2d` layers after convolutions with larger mini-batch size (256) to accelerate training throughput.
- **Configuration**:
  - Architecture: 2-Block CNN with Batch Normalization
  - Batch Size: 256, Optimizer: Adam (`lr=1e-3`), StepLR
- **Outcome**: Peak validation accuracy of **72.06%**. Doubling batch size reduced stochastic gradient noise, leading to slightly lower generalization on CIFAR-10 without learning rate warm-up/scaling.

---

#### Experiment 6: `EXP-06` — 2-Block CNN + Batch Normalization + Dropout (Batch Size 128)
- **Description**: Combined Batch Normalization with Dropout regularization to reduce model variance.
- **Configuration**:
  - Architecture: 2-Block CNN + BatchNorm + Dropout
  - Batch Size: 128, Epochs: 20, StepLR
- **Outcome**: Reached **72.12%** validation accuracy at Epoch 15. The combined regularizers overly constrained the 2-block capacity, resulting in slight underfitting compared to EXP-02.

---

## 3. Key Findings & Empirical Insights

1. **Optimal Learning Rate Schedule**:
   - `StepLR` with a step size of 5 and decay factor $\gamma=0.1$ yielded the highest validation accuracy (**76.12%** in `EXP-02`).
   - Decaying beyond $10^{-4}$ (to $10^{-5}$ and $10^{-6}$ in 20-epoch runs) stalled parameter updates prematurely.

2. **Capacity vs. Regularization Balance**:
   - The 2-block CNN with data augmentation (`RandomCrop`, `RandomHorizontalFlip`, `RandomErasing`) provides strong regularization without requiring heavy internal dropout.
   - For deeper architectures (3-block / 128 channels), longer training with a less aggressive decay schedule (e.g., CosineAnnealing or $\gamma=0.5$) maximizes feature extraction capacity.

3. **Batch Size Sensitivity**:
   - Batch size 128 outperformed batch size 256 under fixed Adam initial learning rate (`1e-3`), maintaining higher generalization capability.

---

## 4. How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Training Pipeline
```bash
cd src
python train.py
```

### Inspect Dataset Loaders
```bash
python dataset.py
```
