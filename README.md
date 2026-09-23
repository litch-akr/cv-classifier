# CIFAR-10 Computer Vision Classifier: Architecture, Experiments & Evaluation

A modular PyTorch implementation of deep convolutional neural network (CNN) architectures and transfer learning models trained and evaluated on the CIFAR-10 dataset (*airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*).

```
CIFAR-10 Pipeline:
[Raw 32x32 RGB Images]
         │
         ▼
[Data Augmentation & Normalization]
  ├─ RandomCrop(32, padding=4)
  ├─ RandomHorizontalFlip()
  └─ Normalization (ImageNet / CIFAR-10 stats)
         │
         ▼
[Model Architecture]
  ├─ Custom Multi-Block CNN (from scratch)
  └─ CIFAR-Adapted Pretrained ResNet-18 (Transfer Learning)
         │
         ▼
[Optimization & Discriminative Fine-Tuning]
         │
         ▼
[Evaluation & Error Analysis]
  ├─ Classification Report (Precision, Recall, F1)
  ├─ Confusion Matrices (Validation & Test)
  └─ Exported JSON Metrics
```

---

## 1. Project Overview & Architecture

### 1.1 Data Pipeline (`src/dataset.py`)

- **Data Partitioning**:
  - **Training Set**: 45,000 images (deterministic split with seed `42`).
  - **Validation Set**: 5,000 images (deterministic split with seed `42`).
  - **Test Set**: 10,000 images (held-out test benchmark).
- **Data Augmentation (Training)**:
  - `RandomHorizontalFlip(p=0.5)`
  - `RandomCrop(32, padding=4)`
  - Optional: `RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))` for custom CNN experiments.
- **Normalization**:
  - ImageNet channel statistics for transfer learning: Mean `(0.485, 0.456, 0.406)`, Std `(0.229, 0.224, 0.225)`.
  - CIFAR-10 channel statistics for baseline models: Mean `(0.4914, 0.4822, 0.4465)`, Std `(0.2470, 0.2435, 0.2616)`.
- **Data Loading**: PyTorch `DataLoader` with pinned memory and non-blocking GPU batch streaming.

---

### 1.2 Model Architectures

#### A. Custom Convolutional Architectures (`src/model.py`)
- **Baseline 2-Block CNN (`CIFARClassifier`)**:
  - **Block 1**: `Conv2d(3, 32, kernel_size=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (Output: $32 \times 16 \times 16$)
  - **Block 2**: `Conv2d(32, 64, kernel_size=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (Output: $64 \times 8 \times 8$)
  - **Classifier**: `Flatten()` $\rightarrow$ `Linear(4096, 512)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(512, 10)`
- **Architectural Variations**:
  - 3-Block CNN ($3 \rightarrow 32 \rightarrow 64 \rightarrow 128$)
  - Batch Normalization (`BatchNorm2d`)
  - Dropout regularization ($p=0.25 / 0.5$)

#### B. CIFAR-Adapted ResNet-18 (`src/resnet_model.py`)
Standard ImageNet ResNet-18 uses a $7 \times 7$ convolution with stride 2 and max pooling, which quickly downsamples $32 \times 32$ inputs to $4 \times 4$ feature maps before intermediate layers. To resolve this:
- **Stem Replacement**: Initial `Conv2d(3, 64, kernel_size=7, stride=2, padding=3)` is replaced by `Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)`.
- **Pooling Bypass**: `maxpool` replaced with `nn.Identity()` to preserve spatial resolution throughout the residual stages.
- **Classification Head**: Final linear layer replaced with `nn.Linear(512, 10)`.
- **Weights**: Initialized with ImageNet pre-trained weights (`ResNet18_Weights.DEFAULT`).

---

### 1.3 Training Strategies (`src/train.py`, `src/train_resnet.py`)

- **Loss Function**: Multi-class Cross-Entropy Loss (`nn.CrossEntropyLoss`).
- **Discriminative Layer-Wise Learning Rates**:
  - Backbone residual layers (`layer1` through `layer4`): $1 \times 10^{-4}$ to avoid catastrophic forgetting.
  - Classification head (`fc`): $1 \times 10^{-3}$ for faster convergence.
- **Model Checkpointing**: Automatic checkpoint saving (`val_acc`, epoch, model weights, optimizer state) based on peak validation accuracy.

---

## 2. Experimental Results & Ablation Studies

### 2.1 Complete Experiment Matrix

| Exp ID | Experiment Name | Architecture / Setup | Input Size | Optimizer & LR Schedule | Epochs | Best Val Acc | Peak Epoch |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **EXP-01** | Baseline 2-Block CNN | 2-Block (3 $\rightarrow$ 32 $\rightarrow$ 64) | $32 \times 32$ | Adam (`lr=1e-3`, constant) | 10 | **75.18%** | Epoch 9 |
| **EXP-02** | 2-Block CNN + StepLR | 2-Block (3 $\rightarrow$ 32 $\rightarrow$ 64) | $32 \times 32$ | Adam + StepLR (`step=5, γ=0.1`) | 10 | **76.12%** | **Epoch 10** |
| **EXP-03** | 2-Block CNN (Extended) | 2-Block (3 $\rightarrow$ 32 $\rightarrow$ 64) | $32 \times 32$ | Adam + StepLR (`step=5, γ=0.1`) | 20 | **72.54%** | Epoch 14 |
| **EXP-04** | 3-Block CNN (Deeper) | 3-Block (3 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 128) | $32 \times 32$ | Adam + StepLR (`step=5, γ=0.1`) | 20 | **74.78%** | Epoch 13 |
| **EXP-05** | 2-Block CNN + BN (Batch 256) | 2-Block + BatchNorm | $32 \times 32$ | Adam + StepLR (`step=5, γ=0.1`) | 20 | **72.06%** | Epoch 17 |
| **EXP-06** | 2-Block CNN + BN + Dropout | 2-Block + BatchNorm + Dropout | $32 \times 32$ | Adam + StepLR (`step=5, γ=0.1`) | 20 | **72.12%** | Epoch 15 |
| **EXP-07** | Frozen ResNet-18 (Upscaled) | Standard ResNet-18 (Frozen Backbone) | $224 \times 224$ | Adam (`lr=1e-3`) | 10 | **91.80%** | Epoch 10 |
| **EXP-08** | CIFAR-Adapted ResNet-18 (Uniform) | Adapted ResNet-18 (Fine-Tuned) | $32 \times 32$ | Adam (`lr=1e-3` uniform) | 10 | **90.48%** | Epoch 9 |
| **EXP-09** | **CIFAR-Adapted ResNet-18 (Discriminative)** | **Adapted ResNet-18 (Fine-Tuned)** | $32 \times 32$ | **Adam (`backbone=1e-4`, `fc=1e-3`)** | **15** | **91.72%** | **Epoch 12** |

---

### 2.2 Detailed Experiment Reports

#### Custom CNN Exploration (EXP-01 – EXP-06)

- **EXP-01 (Baseline 2-Block CNN)**:
  - *Setup*: 2 Conv blocks, Adam optimizer ($10^{-3}$), 10 epochs, batch size 128.
  - *Result*: Achieved **75.18%** validation accuracy at Epoch 9. Overfitting emerged after Epoch 6 as validation loss plateaued around 0.72.
- **EXP-02 (2-Block CNN + StepLR Scheduler)**:
  - *Setup*: Added `StepLR(step_size=5, gamma=0.1)` reducing learning rate to $10^{-4}$ at epoch 6.
  - *Result*: Reached **76.12%** at Epoch 10, delivering the best performance among models trained from scratch.
- **EXP-03 (Extended 20-Epoch Training)**:
  - *Setup*: 20 epochs with StepLR decaying every 5 epochs ($10^{-3} \rightarrow 10^{-4} \rightarrow 10^{-5} \rightarrow 10^{-6}$).
  - *Result*: Peaked at **72.54%** at Epoch 14. Decaying to $10^{-5}$ and $10^{-6}$ caused premature parameter freezing.
- **EXP-04 (3-Block CNN Capacity Expansion)**:
  - *Setup*: 3 Conv blocks ($32 \rightarrow 64 \rightarrow 128$) with StepLR for 20 epochs.
  - *Result*: Reached **74.78%** at Epoch 13, showing better representation learning than EXP-03 over 20 epochs.
- **EXP-05 & EXP-06 (Regularization & Batch Size Ablations)**:
  - *Setup*: BatchNorm and Dropout additions evaluated on 20-epoch schedules.
  - *Result*: Validation accuracy remained in the 72.0%–72.1% range; aggressive regularization constrained the small 2-block model capacity.

---

#### Transfer Learning & Adapted ResNet-18 (EXP-07 – EXP-09)

- **EXP-07 (Frozen Backbone on Upscaled 224x224 Inputs)**:
  - *Setup*: Pretrained ResNet-18 with frozen feature extractor and trained classification head on $224 \times 224$ images.
  - *Result*: Validation accuracy reached **91.80%** (Train Acc: 96.69%, Val Loss: 0.2916). However, high-resolution upscaling introduces significant computational and memory overhead.
- **EXP-08 (CIFAR-Adapted ResNet-18 with Uniform Learning Rate)**:
  - *Setup*: ResNet-18 modified for direct $32 \times 32$ processing ($3 \times 3$ stride-1 stem, removed maxpool). Trained end-to-end with uniform learning rate `1e-3` for 10 epochs.
  - *Result*: Reached **90.48%** validation accuracy (Epoch 9). High initial learning rate across the entire network caused minor gradient instability in early layers.
- **EXP-09 (CIFAR-Adapted ResNet-18 with Discriminative Layer Rates - Best Model)**:
  - *Setup*: Native $32 \times 32$ processing, discriminative learning rates (`1e-4` for residual blocks, `1e-3` for classifier head), trained for 15 epochs.
  - *Training Progress*:
    ```
    Epoch [01/15] Train Loss: 0.9298  Train Acc: 67.26%  Val Loss: 0.5343  Val Acc: 80.40%
    Epoch [03/15] Train Loss: 0.3631  Train Acc: 87.48%  Val Loss: 0.3356  Val Acc: 88.30%
    Epoch [06/15] Train Loss: 0.1952  Train Acc: 93.20%  Val Loss: 0.3024  Val Acc: 90.06%
    Epoch [09/15] Train Loss: 0.1262  Train Acc: 95.52%  Val Loss: 0.3074  Val Acc: 90.92%
    Epoch [10/15] Train Loss: 0.1094  Train Acc: 96.15%  Val Loss: 0.2820  Val Acc: 91.54%
    Epoch [12/15] Train Loss: 0.0883  Train Acc: 96.84%  Val Loss: 0.2962  Val Acc: 91.72% ★ (Best)
    Epoch [15/15] Train Loss: 0.0697  Train Acc: 97.55%  Val Loss: 0.3320  Val Acc: 91.60%
    ```
  - *Result*: Achieved **91.72%** validation accuracy at native $32 \times 32$ resolution with low latency and memory footprint. Checkpoint saved to `src/best_resnet18_cifar32_discriminative.pth`.

---

## 3. Comprehensive Evaluation & Benchmark Metrics

The best checkpoint (`src/best_resnet18_cifar32_discriminative.pth`, Epoch 12) was evaluated on both the validation set (5,000 samples) and the official test set (10,000 samples) using `src/evaluate.py`.

### 3.1 Overall Performance Summary

| Metric | Validation Set (5,000 images) | Test Set (10,000 images) |
| :--- | :---: | :---: |
| **Accuracy** | **91.72%** | **91.54%** |
| **Macro Precision** | 0.9177 | 0.9166 |
| **Macro Recall** | 0.9174 | 0.9154 |
| **Macro F1-Score** | 0.9170 | 0.9155 |
| **Weighted F1-Score** | 0.9175 | 0.9155 |

---

### 3.2 Per-Class Breakdown

| Class | Val Precision | Val Recall | Val F1 | Test Precision | Test Recall | Test F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Airplane** | 0.9078 | 0.9488 | 0.9279 | 0.8925 | 0.9550 | 0.9227 |
| **Automobile** | 0.9762 | 0.9629 | 0.9695 | 0.9648 | 0.9590 | 0.9619 |
| **Bird** | 0.9273 | 0.8628 | 0.8939 | 0.9315 | 0.8570 | 0.8927 |
| **Cat** | 0.7761 | 0.8684 | 0.8196 | 0.8004 | 0.8700 | 0.8337 |
| **Deer** | 0.9002 | 0.9193 | 0.9097 | 0.9092 | 0.9110 | 0.9101 |
| **Dog** | 0.8812 | 0.8230 | 0.8511 | 0.8842 | 0.8320 | 0.8573 |
| **Frog** | 0.9487 | 0.9487 | 0.9487 | 0.9306 | 0.9520 | 0.9412 |
| **Horse** | 0.9476 | 0.9400 | 0.9438 | 0.9475 | 0.9390 | 0.9432 |
| **Ship** | 0.9583 | 0.9563 | 0.9573 | 0.9524 | 0.9410 | 0.9467 |
| **Truck** | 0.9536 | 0.9441 | 0.9488 | 0.9533 | 0.9380 | 0.9456 |

---

### 3.3 Error & Confusion Analysis

1. **Vehicles vs. Animals Distinction**:
   - High discrimination performance was achieved on vehicular categories (*automobile*, *ship*, *truck*), with F1-scores consistently exceeding **94.5%–96.9%**.
2. **Fine-Grained Semantic Overlap**:
   - The primary source of misclassifications occurs between visually similar quadruped mammals (*cat* vs. *dog*). Cat precision is 80.04% and Dog recall is 83.20% on the test set due to pose and background similarities.
   - Another minor confusion path exists between *bird* and *airplane* due to high-altitude backgrounds (sky).
3. **Artifacts Exported**:
   - `outputs/evaluation/confusion_matrix_validation.png`
   - `outputs/evaluation/confusion_matrix_test.png`
   - `outputs/evaluation/metrics.json`

---

## 4. Key Takeaways & Empirical Insights

1. **Pretraining & Transfer Learning Gain**:
   - Moving from scratch CNNs (76.12%) to pretrained ResNet-18 (91.72% val / 91.54% test) yields a **+15.6 percentage point** gain, demonstrating the power of transferred visual representations.
2. **Receptive Field Adaptation for Low-Resolution Inputs**:
   - Adapting ResNet-18 stem ($3\times 3$, stride 1, no max pooling) preserves the $32 \times 32$ spatial dimension across stages, matching the 91.80% accuracy of $224 \times 224$ upscaled inputs while dramatically reducing compute and latency.
3. **Discriminative Learning Rates**:
   - Decoupling the backbone learning rate ($10^{-4}$) from the classification head ($10^{-3}$) prevents catastrophic forgetting of pre-trained filters while allowing the new classifier to adapt swiftly.

---

## 5. Repository Structure

```
cv-classifier/
├── README.md                                    # Main documentation and benchmark report
├── requirements.txt                             # Python dependencies
├── outputs/
│   └── evaluation/
│       ├── confusion_matrix_test.png            # Test set confusion matrix
│       ├── confusion_matrix_validation.png      # Validation set confusion matrix
│       └── metrics.json                         # Serialized evaluation metrics
└── src/
    ├── dataset.py                               # CIFAR-10 data loaders & augmentations
    ├── model.py                                 # Custom CNN implementations
    ├── resnet_model.py                          # CIFAR-adapted ResNet-18 architecture
    ├── train.py                                 # Training pipeline for custom CNN
    ├── train_resnet.py                          # Fine-tuning pipeline for ResNet-18
    ├── evaluate.py                              # Evaluation suite & confusion matrix generator
    ├── best_model.pth                           # Checkpoint: Best Custom CNN (76.12%)
    └── best_resnet18_cifar32_discriminative.pth # Checkpoint: Best Discriminative ResNet-18 (91.72%)
```

---

## 6. How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Test Dataset Loader
```bash
cd src
python dataset.py
```

### Step 3: Train Custom CNN Baseline
```bash
python train.py
```

### Step 4: Train / Fine-Tune CIFAR-Adapted ResNet-18
```bash
python train_resnet.py
```

### Step 5: Run Evaluation & Generate Metrics
```bash
python evaluate.py
```
This evaluates the best checkpoint against both validation and test sets, computes per-class metrics, plots confusion matrices, and writes `outputs/evaluation/metrics.json`.
