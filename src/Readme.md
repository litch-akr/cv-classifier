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
- **Outcome**: Achieved **75.18%** validation accuracy at Epoch 9. Mild signs of overfitting observed around epochs 6–7 as validation loss plateaued.

---

#### Experiment 2: `EXP-02` — Baseline 2-Block CNN + StepLR Scheduler (10 Epochs)
- **Description**: Introduction of `StepLR(step_size=5, gamma=0.1)` to decrease learning rate from $10^{-3}$ to $10^{-4}$ at epoch 6.
- **Configuration**:
  - Architecture: 2 Conv Blocks (3 $\rightarrow$ 32 $\rightarrow$ 64)
  - Optimizer: Adam (`lr=1e-3`), StepLR scheduler
  - Batch Size: 128, Epochs: 10
- **Outcome**: Best overall performance achieved (**76.12%** at Epoch 10). LR reduction allowed finer convergence and reduced validation loss to 0.6807.

---

#### Experiment 3: `EXP-03` — Baseline 2-Block CNN + StepLR (20 Epochs Extended)
- **Description**: Extending training to 20 epochs with `StepLR(step_size=5, gamma=0.1)` decaying every 5 epochs ($10^{-3} \rightarrow 10^{-4} \rightarrow 10^{-5} \rightarrow 10^{-6}$).
- **Configuration**:
  - Architecture: 2 Conv Blocks (3 $\rightarrow$ 32 $\rightarrow$ 64)
  - Optimizer: Adam (`lr=1e-3`), StepLR (`step_size=5, gamma=0.1`)
  - Batch Size: 128, Epochs: 20
- **Outcome**: Best validation accuracy was **72.54%** at Epoch 14. Decaying learning rate too aggressively ($10^{-5}$ and $10^{-6}$) caused premature parameter freeze, preventing further optimization gains.

---

#### Experiment 4: `EXP-04` — 3-Block CNN (Deeper Architecture: 3 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 128)
- **Description**: Adding a third convolutional block to evaluate whether expanded model capacity yields better feature representations.
- **Configuration**:
  - Architecture: 3 Conv Blocks (3 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 128)
  - Optimizer: Adam (`lr=1e-3`), StepLR (`step_size=5, gamma=0.1`)
  - Batch Size: 128, Epochs: 20
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
