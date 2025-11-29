# EEEM066 Knife Classification Project - Complete Codebase Documentation

**Author:** Student Documentation  
**Project:** Knife Classification Using Deep Learning  
**Module:** EEEM066 - Fundamentals of Machine Learning  
**Date:** November 2025

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Coursework Requirements for Distinction](#2-coursework-requirements-for-distinction)
3. [Architecture & System Flow](#3-architecture--system-flow)
4. [Entry Points & Execution Flow](#4-entry-points--execution-flow)
5. [File-by-File Breakdown](#5-file-by-file-breakdown)
6. [Data Pipeline](#6-data-pipeline)
7. [Training Pipeline](#7-training-pipeline)
8. [Testing Pipeline](#8-testing-pipeline)
9. [Hyperparameter Tuning Strategy](#9-hyperparameter-tuning-strategy)
10. [Code Snippets with Line Numbers](#10-code-snippets-with-line-numbers)
11. [Execution Instructions](#11-execution-instructions)

---

## 1. Project Overview

### Problem Statement
Develop an AI-based weapon analysis system to identify knife types from photographs. The UK has seen a 34% increase in knife crimes since 2010/11, with ~45,000 offenses in 2021/22.

### Solution
A deep learning image classification pipeline using:
- **Pre-trained CNN models** (EfficientNet, ResNet, etc.)
- **Transfer learning** for 543 knife classes
- **Data augmentation** for robustness
- **Hyperparameter tuning** for optimal performance

### Key Metrics
- **mAP@5** (mean Average Precision at top-5): Primary evaluation metric
- **Top-1 Accuracy**: Single best prediction accuracy
- **Top-5 Accuracy**: Top 5 predictions contain correct class

---

## 2. Coursework Requirements for Distinction

### Assessment Breakdown (Total: 100 marks)

#### Section 1: Baseline & Data Augmentation (30 marks)
1. **Baseline Model** (10 marks)
   - Train with minimal augmentation
   - Report mAP@5, training time
   - Analyze results

2. **Data Augmentation Experiments** (20 marks)
   - Test 3+ augmentation techniques
   - Compare performance improvements
   - Explain why augmentations work
   - Present results in tables/plots

#### Section 2: Architecture Comparison (30 marks)
1. **Multiple Architectures** (15 marks)
   - Test 3+ different CNN architectures
   - Compare: EfficientNet, ResNet, MobileNet, etc.
   - Analyze trade-offs (accuracy vs. speed)

2. **Results & Analysis** (15 marks)
   - Professional tables and plots
   - Deep analysis of architecture implications
   - Computational efficiency discussion

#### Section 3: Learning Rate & Batch Size (30 marks)
1. **Learning Rate Experiments** (15 marks)
   - Test multiple learning rates
   - Use learning rate schedulers
   - Document convergence behavior

2. **Batch Size Experiments** (15 marks)
   - Test different batch sizes
   - Analyze memory vs. performance
   - Discuss optimal configurations

#### Report Quality (10 marks)
- Professional presentation
- Clear tables, plots, and diagrams
- Proper citations (Harvard/IEEE)
- Word limit: 200 words per section (excluding visuals)

### Key Success Criteria for Distinction (70%+)
✅ **Systematic experimentation** with well-documented results  
✅ **Deep analysis** of why techniques work/fail  
✅ **Professional presentation** with clear visuals  
✅ **Comprehensive comparison** across all experiments  
✅ **Critical thinking** about trade-offs and limitations  
✅ **Reproducible results** with proper seed setting  

---

## 3. Architecture & System Flow

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    KNIFE CLASSIFICATION SYSTEM                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│   Dataset       │
│  CSV Files      │
│ ┌─────────────┐ │
│ │ train.csv   │ │
│ │ validation  │ │
│ │ test.csv    │ │
│ └─────────────┘ │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────┐        │
│  │ Load Images  │──▶│  Transform   │──▶│   DataLoader  │        │
│  │ (data.py)    │   │ & Augment    │   │  (batching)   │        │
│  └──────────────┘   └──────────────┘   └───────────────┘        │
│   knifeDataset class                                            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│   TRAINING PIPELINE       │   │   TESTING PIPELINE        │
│  (Training.py)            │   │   (Testing.py)            │
│                           │   │                           │
│  ┌────────────────────┐   │   │  ┌────────────────────┐   │
│  │ Load Pretrained    │   │   │  │ Load Trained Model │   │
│  │ Model (timm)       │   │   │  │ from Checkpoint    │   │
│  └──────────┬─────────┘   │   │  └──────────┬─────────┘   │
│             │             │   │             │             │
│             ▼             │   │             ▼             │
│  ┌────────────────────┐   │   │  ┌────────────────────┐   │
│  │ Training Loop      │   │   │  │ Evaluation Loop    │   │
│  │ - Forward pass     │   │   │  │ - Forward pass     │   │
│  │ - Loss calculation │   │   │  │ - Metric calc      │   │
│  │ - Backward pass    │   │   │  │ - mAP@5 reporting  │   │
│  │ - Optimizer step   │   │   │  └────────────────────┘   │
│  │ - Validation       │   │   │                           │
│  └────────────────────┘   │   └───────────────────────────┘
│             │             │
│             ▼             │
│  ┌────────────────────┐   │
│  │ Save Checkpoints   │   │
│  │ (per epoch)        │   │
│  └────────────────────┘   │
└───────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     SUPPORTING MODULES                          │
│                                                                 │
│  args.py         - Command-line argument parsing                │
│  utils.py        - Helper functions (metrics, logging, etc.)    │
│  src/            - Modular components                           │
│    ├── optimizers.py      - Optimizer initialization            │
│    ├── lr_schedulers.py   - Learning rate schedulers            │
│    └── transforms.py      - Custom augmentation transforms      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          OUTPUT                                 │
│                                                                 │
│  logs/           - Training/testing logs with timestamps        │
│  checkpoints/    - Saved model weights per epoch                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Entry Points & Execution Flow

### Primary Entry Points

#### 1. **Training Entry Point: `Training.py`**

**Purpose:** Train a knife classification model from scratch

**Execution Flow:**
```python
# Line 1-23: Imports and global argument parsing
# Line 83-141: main() function - orchestrates entire training

main() execution sequence:
├─ Line 86-87:  Set random seed for reproducibility
├─ Line 89-95:  Load training & validation datasets
├─ Line 97-98:  Initialize device (GPU/CPU)
├─ Line 100-101: Create model from timm library
├─ Line 103-105: Initialize optimizer & scheduler
├─ Line 107-108: Setup loss function & mixed precision scaler
├─ Line 110-121: Setup logging system
├─ Line 124-127: Main training loop (iterate epochs)
│   ├─ Line 126: Train for one epoch
│   ├─ Line 127: Validate on validation set
│   └─ Line 129-134: Save model checkpoint
└─ End
```

**Call:**
```bash
python Training.py --model_mode tf_efficientnet_b0 \
                   --dataset_location ../EEEM066_KnifeHunter \
                   --epochs 10 --batch_size 32 --learning_rate 0.00005
```

#### 2. **Testing Entry Point: `Testing.py`**

**Purpose:** Evaluate a trained model on test dataset

**Execution Flow:**
```python
# Line 1-23: Imports and global argument parsing
# Line 48-90: main() function - orchestrates testing

main() execution sequence:
├─ Line 51-52:  Set random seed
├─ Line 54-57:  Load test dataset
├─ Line 59-60:  Initialize device
├─ Line 62-65:  Load trained model from checkpoint
├─ Line 67-80:  Setup logging
├─ Line 82-83:  Run evaluation
└─ Line 84:     Print completion message
```

**Call:**
```bash
python Testing.py --model_mode tf_efficientnet_b0 \
                  --model-path Knife-Effb0/model.pth \
                  --dataset_location ../EEEM066_KnifeHunter
```

---

## 5. File-by-File Breakdown

### Core Training/Testing Files

#### **Training.py** (141 lines)

**Purpose:** Main training script with training and validation loops

**Key Functions:**

1. **`train()`** (Lines 27-49)
   ```python
   def train(train_loader, model, criterion, optimizer, scaler, scheduler, epoch, valid_accuracy, start, log):
   ```
   - Iterates through training batches
   - Performs forward pass with mixed precision
   - Computes loss and backpropagates
   - Updates weights using optimizer
   - Steps learning rate scheduler
   - Logs training progress
   - **Returns:** `[average_loss]`

2. **`evaluate()`** (Lines 51-73)
   ```python
   def evaluate(val_loader, model, criterion, epoch, train_loss, start, log):
   ```
   - Sets model to eval mode
   - Performs inference on validation set
   - Computes mAP@5 metric
   - Logs validation results
   - **Returns:** `[mAP@5_score]`

3. **`main()`** (Lines 75-141)
   - Entry point for training
   - Orchestrates entire training pipeline
   - Saves checkpoints after each epoch

**Key Features:**
- ✅ Mixed precision training (AMP) for faster computation
- ✅ Automatic checkpoint saving per epoch
- ✅ Detailed logging with timestamps
- ✅ Seed setting for reproducibility

---

#### **Testing.py** (93 lines)

**Purpose:** Evaluate trained models on test set

**Key Functions:**

1. **`evaluate()`** (Lines 27-46)
   ```python
   def evaluate(test_loader, model, start, log):
   ```
   - Similar to Training.py evaluate but for test set
   - No criterion needed (no loss computation)
   - Computes final mAP@5 on test data
   - **Returns:** `[mAP@5_score]`

2. **`main()`** (Lines 48-90)
   - Loads trained model weights
   - Runs evaluation on test set
   - Logs results with UUID for tracking

**Key Differences from Training:**
- No training loop
- Loads pre-trained weights via `model.load_state_dict()`
- Only forward passes (no backpropagation)
- Simpler logging format

---

### Configuration & Arguments

#### **args.py** (145 lines)

**Purpose:** Command-line argument parsing and configuration management

**Structure:**

1. **`argument_parser()`** (Lines 4-115)
   - Returns ArgumentParser with all configuration options
   
   **Argument Categories:**
   
   a) **Model & Data** (Lines 6-18)
   ```python
   --model_mode: str = 'tf_efficientnet_b0'  # Model architecture
   --dataset_location: str                    # Path to images
   --train_datacsv: str = 'dataset/train.csv'
   --val_datacsv: str = 'dataset/validation.csv'
   --test_datacsv: str = 'dataset/test.csv'
   --n_classes: int = 543                     # Number of knife classes
   --num_workers: int = 8                     # DataLoader workers
   --seed: int = None                         # Random seed
   --resized_img_weight: int = 224            # Image width
   --resized_img_height: int = 224            # Image height
   ```

   b) **Training Hyperparameters** (Lines 22-24)
   ```python
   --epochs: int = 10           # Number of training epochs
   --batch_size: int = 32       # Batch size
   ```

   c) **Data Augmentation** (Lines 28-42)
   ```python
   --brightness: float = 0.2          # ColorJitter brightness
   --contrast: float = 0.2            # ColorJitter contrast
   --saturation: float = 0.2          # ColorJitter saturation
   --hue: float = 0.2                 # ColorJitter hue
   --random_rotation: int = 0         # Max rotation degrees
   --vertical_flip: float = 0         # Vertical flip probability
   --horizontal_flip: float = 0       # Horizontal flip probability
   --random-erase: action             # Enable random erasing
   --color-aug: action                # Enable color augmentation
   ```

   d) **Optimization** (Lines 46-87)
   ```python
   --optim: str = 'adam'              # Optimizer choice
   --learning_rate: float = 0.0003    # Initial learning rate
   --weight-decay: float = 5e-04      # L2 regularization
   --momentum: float = 0.9            # SGD momentum
   --adam-beta1: float = 0.9          # Adam β1
   --adam-beta2: float = 0.999        # Adam β2
   ```

   e) **Learning Rate Scheduler** (Lines 91-113)
   ```python
   --lr-scheduler: str = 'multi_step'     # Scheduler type
   --stepsize: List[int] = [20, 40]       # Decay steps
   --gamma: float = 0.1                   # Decay factor
   ```

2. **`optimizer_kwargs()`** (Lines 117-131)
   - Converts parsed args to optimizer initialization dict
   - **Returns:** Dict with optimizer parameters

3. **`lr_scheduler_kwargs()`** (Lines 133-145)
   - Converts parsed args to scheduler initialization dict
   - **Returns:** Dict with scheduler parameters

---

### Data Handling

#### **data.py** (78 lines)

**Purpose:** Dataset class for loading and preprocessing knife images

**Key Class:**

**`knifeDataset(Dataset)`** (Lines 18-78)

Inherits from `torch.utils.data.Dataset`

**Methods:**

1. **`__init__(self, images_df, mode="train")`** (Lines 19-22)
   ```python
   # Line 19-22
   def __init__(self, images_df, mode="train"):
       self.images_df = images_df.copy()
       self.mode = mode
       self.transforms = self.build_transforms()
   ```
   - `images_df`: pandas DataFrame with columns [Id, Label]
   - `mode`: "train", "val", or "test"

2. **`__len__(self)`** (Lines 24-25)
   - Returns total number of images

3. **`build_transforms(self)`** (Lines 28-50)
   ```python
   # Line 28-50: Critical for data augmentation
   def build_transforms(self):
       if self.mode == "train":
           transform_list = [
               T.Resize((224, 224)),
               T.ColorJitter(...) if args.brightness > 0 else None,
               T.RandomRotation(...) if args.random_rotation > 0 else None,
               T.RandomVerticalFlip(p=...) if args.vertical_flip > 0 else None,
               T.RandomHorizontalFlip(p=...) if args.horizontal_flip > 0 else None,
               T.ToTensor(),
               ColorAugmentation() if args.color_aug else None,
               RandomErasing() if args.random_erase else None,
               T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
           ]
           # Remove None entries
           return T.Compose([t for t in transform_list if t is not None])
   ```
   
   **Training Transforms:**
   - Resize to (224, 224)
   - Color jittering (brightness, contrast, saturation, hue)
   - Random rotation
   - Random flips (horizontal/vertical)
   - ToTensor conversion
   - Custom color augmentation (PCA-based)
   - Random erasing
   - ImageNet normalization
   
   **Validation/Test Transforms:**
   - Resize to (224, 224)
   - ToTensor conversion
   - ImageNet normalization only

4. **`__getitem__(self, index)`** (Lines 52-60)
   ```python
   # Line 52-60
   def __getitem__(self, index):
       X, fname = self.read_images(index)
       labels = self.images_df.iloc[index].Label
       X = self.transforms(X)
       return X.float(), labels, fname
   ```
   - Loads single image and label
   - Applies transforms
   - **Returns:** (image_tensor, label, filename)

5. **`read_images(self, index)`** (Lines 62-70)
   ```python
   # Line 62-70
   def read_images(self, index):
       filename = str(self.images_df.iloc[index].Id)
       filename_path = os.path.join(args.dataset_location, filename)
       image = cv2.imread(filename_path)
       image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       image = Image.fromarray(image)
       return image, filename
   ```
   - Reads image from disk
   - Converts BGR (OpenCV) to RGB
   - Converts to PIL Image
   - **Returns:** (PIL_image, filename)

**Integration:**
```python
# Used in Training.py (Lines 89-95)
train_imlist = pd.read_csv(args.train_datacsv)
train_gen = knifeDataset(train_imlist, mode="train")
train_loader = DataLoader(train_gen, batch_size=32, shuffle=True, ...)
```

---

### Utilities & Helpers

#### **utils.py** (178 lines)

**Purpose:** Helper functions for metrics, logging, and utilities

**Key Components:**

1. **`AverageMeter`** (Lines 15-30)
   ```python
   # Lines 15-30
   class AverageMeter(object):
       def __init__(self): ...
       def reset(self): ...
       def update(self, val, n=1):
           self.sum += val * n
           self.count += n
           self.avg = self.sum / self.count
   ```
   - Tracks running average of metrics (loss, mAP)
   - Used in training/validation loops

2. **`Logger`** (Lines 34-62)
   ```python
   # Lines 34-62
   class Logger(object):
       def open(self, file, mode='w'): ...
       def write(self, message, is_terminal=1, is_file=1): ...
   ```
   - Dual output: terminal + log file
   - Used for experiment tracking

3. **`FocalLoss`** (Lines 65-78)
   - Custom loss function (not used in current implementation)
   - Can replace CrossEntropyLoss for imbalanced classes

4. **`ArcFaceLoss`** (Lines 81-113)
   - Angular margin loss (not used currently)
   - Useful for metric learning

5. **`map_accuracy()`** (Lines 147-158)
   ```python
   # Lines 147-158: Critical evaluation metric
   def map_accuracy(probs, truth, k=5):
       value, top = probs.topk(k, dim=1, largest=True, sorted=True)
       correct = top.eq(truth.view(-1, 1).expand_as(top))
       
       # Calculate mAP@5
       map5 = correct[0]/1 + correct[1]/2 + correct[2]/3 + correct[3]/4 + correct[4]/5
       acc1 = correct[0]  # Top-1 accuracy
       acc5 = sum(correct[0:5])  # Top-5 accuracy
       return map5, acc1, acc5
   ```
   - **mAP@5 formula:** 
     $$\text{mAP@5} = \frac{1}{5}\sum_{i=1}^{5}\frac{\text{correct}_i}{i}$$
   - **Returns:** (mAP@5, top-1 acc, top-5 acc)

6. **`format_log_message()`** (Lines 162-163)
   ```python
   # Line 162
   def format_log_message(mode, i, epoch, loss, mAP, time_str):
       return f'| {mode:<5} | {i:5.1f} | {epoch:5.1f} | {loss:8.3f} | {mAP:7.3f} | {time_str:<12} |'
   ```
   - Consistent log formatting

7. **`set_seed()`** (Lines 169-178)
   ```python
   # Lines 169-178: Essential for reproducibility
   def set_seed(seed_value):
       random.seed(seed_value)
       np.random.seed(seed_value)
       torch.manual_seed(seed_value)
       torch.cuda.manual_seed(seed_value)
       torch.cuda.manual_seed_all(seed_value)
   ```
   - Sets seeds for all random number generators
   - Ensures reproducible results

---

### Modular Components (src/)

#### **src/optimizers.py** (86 lines)

**Purpose:** Initialize various optimizers with different configurations

**Main Function:**

**`init_optimizer()`** (Lines 7-86)
```python
def init_optimizer(
    model,
    optim="adam",
    lr=0.003,
    weight_decay=5e-4,
    momentum=0.9,
    ...
):
```

**Supported Optimizers:**

1. **Adam** (Lines 47-52)
   ```python
   if optim == "adam":
       return torch.optim.Adam(
           param_groups, lr=lr, weight_decay=weight_decay,
           betas=(adam_beta1, adam_beta2)
       )
   ```
   - Default choice
   - Adaptive learning rate
   - Suitable for most tasks

2. **AMSGrad** (Lines 54-60)
   - Variant of Adam
   - Better convergence guarantees

3. **SGD** (Lines 62-69)
   ```python
   elif optim == "sgd":
       return torch.optim.SGD(
           param_groups, lr=lr, momentum=momentum,
           weight_decay=weight_decay, dampening=..., nesterov=...
       )
   ```
   - Momentum-based optimizer
   - Can use Nesterov acceleration

4. **RMSprop** (Lines 71-77)
   - Adaptive learning rate
   - Good for RNNs

**Staged Learning Rate Support** (Lines 19-42):
- Can set different learning rates for base vs. new layers
- Useful for fine-tuning

---

#### **src/lr_schedulers.py** (30 lines)

**Purpose:** Learning rate scheduling strategies

**Main Function:**

**`init_lr_scheduler()`** (Lines 6-30)

**Supported Schedulers:**

1. **Single Step** (Lines 13-16)
   ```python
   if lr_scheduler == "single_step":
       return torch.optim.lr_scheduler.StepLR(
           optimizer, step_size=stepsize[0], gamma=gamma
       )
   ```
   - Decays LR by gamma at single step

2. **Multi Step** (Lines 18-21)
   ```python
   elif lr_scheduler == "multi_step":
       return torch.optim.lr_scheduler.MultiStepLR(
           optimizer, milestones=stepsize, gamma=gamma
       )
   ```
   - Decays LR at multiple milestones
   - Example: `stepsize=[20, 40]`, `gamma=0.1`
     - Epoch 0-19: lr = initial_lr
     - Epoch 20-39: lr = initial_lr * 0.1
     - Epoch 40+: lr = initial_lr * 0.01

3. **Cosine Annealing** (Lines 22-25)
   ```python
   elif lr_scheduler == "CosineAnnealingLR":
       return torch.optim.lr_scheduler.CosineAnnealingLR(
           optimizer, T_max=T_max, eta_min=0, last_epoch=-1
       )
   ```
   - Smoothly decreases LR following cosine curve
   - Formula: 
     $$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{T_{cur}}{T_{max}}\pi))$$

---

#### **src/transforms.py** (126 lines)

**Purpose:** Custom data augmentation transforms

**Key Classes:**

1. **`Random2DTranslation`** (Lines 9-42)
   ```python
   class Random2DTranslation:
       def __init__(self, height, width, p=0.5):
           # Enlarges image by 12.5% then random crops
   ```
   - Simulates translation/cropping
   - Not currently used in data.py

2. **`RandomErasing`** (Lines 45-98)
   ```python
   class RandomErasing:
       def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
           # Randomly erases rectangular regions
   ```
   - **Purpose:** Occlusion robustness
   - Erases 2%-40% of image area
   - Fills with mean values
   - **When activated:** `args.random_erase = True`

3. **`ColorAugmentation`** (Lines 101-126)
   ```python
   class ColorAugmentation:
       def __init__(self, p=0.5):
           # PCA-based color augmentation (AlexNet-style)
           self.eig_vec = torch.Tensor([...])  # RGB eigenvectors
           self.eig_val = torch.Tensor([...])  # Eigenvalues
   ```
   - **Purpose:** Color variation robustness
   - Adds noise in PCA color space
   - Based on Krizhevsky et al. (ImageNet 2012)
   - **When activated:** `args.color_aug = True`

**Usage in data.py:**
```python
# Line 39-40 in data.py
ColorAugmentation() if args.color_aug else None,
RandomErasing() if args.random_erase else None,
```

---

## 6. Data Pipeline

### Dataset Structure

```
EEEM066_KnifeHunter/          # Root dataset directory
├── Train/                     # Training images
│   ├── BForce1/
│   │   ├── BForce1_1.jpg
│   │   ├── BForce1_2.jpg
│   │   └── ...
│   ├── BForce2/
│   └── ...
├── Validation/                # Validation images
│   └── (similar structure)
└── Test/                      # Test images
    └── (similar structure)

dataset/                       # CSV metadata files
├── classes.csv                # Label to class name mapping
│   # Format: Labels,Classes
│   #         0,BForce1
│   #         1,BForce2
├── train.csv                  # Training set metadata
│   # Format: Id,Label
│   #         Train/BForce1/BForce1_1.jpg,0
├── validation.csv             # Validation set metadata
└── test.csv                   # Test set metadata
```

### Data Loading Process

**Step 1: CSV Loading**
```python
# Training.py Line 89
train_imlist = pd.read_csv(args.train_datacsv)
# Creates DataFrame:
#    Id                              Label
#    Train/BForce1/BForce1_1.jpg     0
#    Train/BForce1/BForce1_2.jpg     0
```

**Step 2: Dataset Instantiation**
```python
# Training.py Line 90
train_gen = knifeDataset(train_imlist, mode="train")
# Initializes knifeDataset with DataFrame
# Sets up train-specific transforms
```

**Step 3: DataLoader Creation**
```python
# Training.py Lines 91-92
train_loader = DataLoader(
    train_gen,
    batch_size=args.batch_size,     # e.g., 32
    shuffle=True,                    # Randomize order
    pin_memory=True,                 # Faster GPU transfer
    num_workers=args.num_workers     # Parallel loading (8)
)
```

**Step 4: Batch Retrieval**
```python
# Training.py Line 30
for i, (images, target, fnames) in enumerate(train_loader):
    # images: Tensor [batch_size, 3, 224, 224]
    # target: Tensor [batch_size] - class labels
    # fnames: List[str] - filenames
```

### Transform Pipeline

**Training Mode:**
```
Input Image (variable size)
    ↓
Resize to 224x224
    ↓
ColorJitter (if enabled)
    ├─ Brightness variation: ±20%
    ├─ Contrast variation: ±20%
    ├─ Saturation variation: ±20%
    └─ Hue variation: ±20%
    ↓
Random Rotation (if enabled)
    └─ Rotate by 0 to max_degrees
    ↓
Random Vertical Flip (if enabled)
    └─ Flip with probability p
    ↓
Random Horizontal Flip (if enabled)
    └─ Flip with probability p
    ↓
ToTensor (convert PIL to Tensor)
    └─ Range: [0, 255] → [0.0, 1.0]
    ↓
Color Augmentation (if enabled)
    └─ PCA-based color jittering
    ↓
Random Erasing (if enabled)
    └─ Erase 2-40% of image area
    ↓
Normalize (ImageNet statistics)
    ├─ Mean: [0.485, 0.456, 0.406]
    └─ Std:  [0.229, 0.224, 0.225]
    ↓
Output Tensor: [3, 224, 224]
```

**Validation/Test Mode:**
```
Input Image
    ↓
Resize to 224x224
    ↓
ToTensor
    ↓
Normalize (ImageNet)
    ↓
Output Tensor: [3, 224, 224]
```

---

## 7. Training Pipeline

### Complete Training Flow

```
main() - Training.py Line 83
    ↓
1. Set Seed (Line 86-87)
    ├─ set_seed(args.seed)
    └─ Ensures reproducibility
    ↓
2. Load Data (Line 89-95)
    ├─ Read train.csv → train_loader
    └─ Read validation.csv → val_loader
    ↓
3. Initialize Model (Line 100-101)
    ├─ model = timm.create_model(args.model_mode, pretrained=True, num_classes=543)
    └─ Load pretrained weights (ImageNet)
    ↓
4. Setup Training Components (Line 103-108)
    ├─ optimizer = init_optimizer(...)
    ├─ scheduler = init_lr_scheduler(...)
    ├─ criterion = nn.CrossEntropyLoss()
    └─ scaler = torch.cuda.amp.GradScaler()  # Mixed precision
    ↓
5. Setup Logging (Line 110-123)
    └─ Create timestamped log file
    ↓
6. Training Loop (Line 126-134) - FOR EACH EPOCH
    ↓
    ├─ 6a. Train Phase (Line 126)
    │   └─ train_metrics = train(...)
    │       ├─ Iterate batches
    │       ├─ Forward pass
    │       ├─ Compute loss
    │       ├─ Backward pass
    │       ├─ Update weights
    │       └─ Step scheduler
    │
    ├─ 6b. Validation Phase (Line 127)
    │   └─ val_metrics = evaluate(...)
    │       ├─ Iterate batches
    │       ├─ Forward pass (no grad)
    │       ├─ Compute mAP@5
    │       └─ Log results
    │
    └─ 6c. Save Checkpoint (Line 129-134)
        └─ torch.save(model.state_dict(), path)
    ↓
END
```

### Training Function Details

**`train()` Function - Lines 27-49**

```python
def train(train_loader, model, criterion, optimizer, scaler, scheduler, epoch, valid_accuracy, start, log):
    model.train()  # Enable dropout, batchnorm training mode
    losses = AverageMeter()
    
    for i, (images, target, fnames) in enumerate(train_loader):
        # 1. Move data to GPU
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # 2. Zero gradients
        optimizer.zero_grad()
        
        # 3. Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            logits = model(images)           # [batch_size, 543]
            loss = criterion(logits, target) # CrossEntropyLoss
        
        # 4. Track metrics
        losses.update(loss.item(), images.size(0))
        
        # 5. Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # 6. Optimizer step with gradient scaling
        scaler.step(optimizer)
        scaler.update()
        
        # 7. Learning rate schedule step (per batch)
        scheduler.step()
        
        # 8. Print progress
        message = format_log_message(...)
        print(f'\r{message}', end='', flush=True)
    
    # 9. Log epoch summary
    log.write(message_train_epoch)
    
    return [losses.avg]
```

**Key Features:**
- ✅ **Mixed Precision Training:** Uses `torch.cuda.amp` for faster training
- ✅ **Per-batch LR scheduling:** `scheduler.step()` called every batch
- ✅ **Gradient scaling:** Prevents underflow in FP16
- ✅ **Non-blocking transfer:** `cuda(non_blocking=True)` for efficiency

### Validation Function Details

**`evaluate()` Function - Lines 51-73**

```python
def evaluate(val_loader, model, criterion, epoch, train_loss, start, log):
    model.eval()  # Disable dropout, batchnorm uses running stats
    map = AverageMeter()
    
    with torch.no_grad():  # No gradient computation
        for i, (images, target, fnames) in enumerate(val_loader):
            # 1. Move data to GPU
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            # 2. Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                logits = model(images)        # [batch_size, 543]
                preds = logits.softmax(1)     # Convert to probabilities
            
            # 3. Compute metrics
            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, target)
            map.update(valid_map5, images.size(0))
            
            # 4. Print progress
            message_val = format_log_message(...)
            print(f'\r{message_val}', end='', flush=True)
        
        # 5. Log epoch summary
        log.write(message_val_epoch)
    
    return [map.avg]
```

**Key Differences from Training:**
- No gradient computation (`torch.no_grad()`)
- Model in eval mode (`model.eval()`)
- Uses mAP@5 instead of loss
- No optimizer or scheduler updates

---

## 8. Testing Pipeline

### Testing Flow

```
main() - Testing.py Line 48
    ↓
1. Set Seed (Line 51-52)
    └─ Reproducible evaluation
    ↓
2. Load Test Data (Line 54-57)
    └─ test_loader from test.csv
    ↓
3. Load Trained Model (Line 62-65)
    ├─ model = timm.create_model(...)
    ├─ model.load_state_dict(torch.load(args.model_path))
    └─ model.to(device)
    ↓
4. Setup Logging (Line 67-80)
    └─ Create timestamped log file
    ↓
5. Evaluate (Line 82)
    └─ test_metrics = evaluate(test_loader, model, timer(), log)
        ├─ Forward passes on all test batches
        ├─ Compute mAP@5
        └─ Log final result
    ↓
6. Print Completion (Line 84)
    └─ "Evaluation complete."
    ↓
END
```

### Model Loading

```python
# Testing.py Lines 62-65
model = timm.create_model(args.model_mode, pretrained=True, num_classes=543)
model.load_state_dict(torch.load(args.model_path))
model.to(device)
```

**Important Notes:**
1. Must use same `model_mode` as training
2. `pretrained=True` creates same architecture (weights overwritten)
3. `load_state_dict()` loads trained weights from checkpoint
4. Model automatically set to eval mode in `evaluate()`

---

## 9. Hyperparameter Tuning Strategy

### Sequential Tuning Approach (For Distinction Grade)

#### Phase 1: Baseline Establishment

**Goal:** Establish baseline performance with minimal augmentation

**Configuration:**
```bash
python Training.py \
    --model_mode tf_efficientnet_b0 \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 0.0003 \
    --seed 42 \
    --brightness 0 \
    --contrast 0 \
    --horizontal_flip 0 \
    --optim adam \
    --lr-scheduler multi_step
```

**Document:**
- Final mAP@5 on validation set
- Training time per epoch
- Total parameters

---

#### Phase 2: Data Augmentation Experiments

**Goal:** Identify best augmentation combinations

**Experiments:**

| Exp # | Augmentations | Expected Impact |
|-------|---------------|-----------------|
| 2.1 | Color Jitter (brightness=0.2, contrast=0.2) | Robustness to lighting |
| 2.2 | Horizontal Flip (p=0.5) | Geometric invariance |
| 2.3 | Random Rotation (15°) | Rotation invariance |
| 2.4 | Color Aug (PCA-based) | Illumination robustness |
| 2.5 | Random Erasing (p=0.5) | Occlusion robustness |
| 2.6 | Combined (best from above) | Cumulative improvement |

**Commands:**
```bash
# Experiment 2.1: Color Jitter
python Training.py --brightness 0.2 --contrast 0.2 --saturation 0.2 --hue 0.2

# Experiment 2.2: Horizontal Flip
python Training.py --horizontal_flip 0.5

# Experiment 2.3: Random Rotation
python Training.py --random_rotation 15

# Experiment 2.4: Color Augmentation
python Training.py --color-aug

# Experiment 2.5: Random Erasing
python Training.py --random-erase

# Experiment 2.6: Combined
python Training.py --brightness 0.2 --horizontal_flip 0.5 --random-erase
```

**Analysis Questions:**
- Which augmentation provides largest improvement?
- Are there diminishing returns with multiple augmentations?
- Does training take longer with augmentation?

---

#### Phase 3: Architecture Comparison

**Goal:** Compare different CNN architectures

**Architectures to Test:**

| Model | Parameters | Input Size | Speed |
|-------|------------|------------|-------|
| tf_efficientnet_b0 | 5.3M | 224x224 | Fast |
| tf_efficientnet_b3 | 12M | 300x300 | Medium |
| resnet50 | 25.6M | 224x224 | Medium |
| resnet101 | 44.5M | 224x224 | Slow |
| mobilenetv3_large_100 | 5.4M | 224x224 | Fastest |

**Commands:**
```bash
# EfficientNet-B0 (baseline)
python Training.py --model_mode tf_efficientnet_b0 --resized_img_weight 224

# EfficientNet-B3
python Training.py --model_mode tf_efficientnet_b3 --resized_img_weight 300 --resized_img_height 300

# ResNet-50
python Training.py --model_mode resnet50

# ResNet-101
python Training.py --model_mode resnet101

# MobileNetV3
python Training.py --model_mode mobilenetv3_large_100
```

**Comparison Metrics:**
- mAP@5 on validation set
- Training time per epoch
- Inference speed
- Model size (MB)
- Memory consumption

---

#### Phase 4: Learning Rate Experiments

**Goal:** Find optimal learning rate and scheduler

**4.1 Learning Rate Sweep:**

| Exp | Learning Rate | Expected Behavior |
|-----|---------------|-------------------|
| 4.1a | 1e-5 | Very slow convergence |
| 4.1b | 5e-5 | Stable, slower convergence |
| 4.1c | 1e-4 | Good convergence |
| 4.1d | 3e-4 | Fast convergence |
| 4.1e | 1e-3 | Potentially unstable |

**Commands:**
```bash
python Training.py --learning_rate 0.00001  # 1e-5
python Training.py --learning_rate 0.00005  # 5e-5
python Training.py --learning_rate 0.0001   # 1e-4
python Training.py --learning_rate 0.0003   # 3e-4
python Training.py --learning_rate 0.001    # 1e-3
```

**4.2 Scheduler Comparison:**

| Scheduler | Description | Best For |
|-----------|-------------|----------|
| multi_step | Step decay at milestones | Stable training |
| single_step | Single step decay | Simple experiments |
| CosineAnnealingLR | Smooth cosine decay | Smooth convergence |

**Commands:**
```bash
# Multi-step (decay at epoch 20, 40)
python Training.py --lr-scheduler multi_step --stepsize 20 40 --gamma 0.1

# Cosine Annealing
python Training.py --lr-scheduler CosineAnnealingLR

# Single step (decay at epoch 30)
python Training.py --lr-scheduler single_step --stepsize 30 --gamma 0.1
```

---

#### Phase 5: Batch Size Experiments

**Goal:** Understand batch size impact

**Experiments:**

| Batch Size | GPU Memory | Training Speed | Generalization      |
|------------|------------|----------------|---------------------|
| 16         | Low        | Slow           | Better (more noise) |
| 32         | Medium     | Medium         | Good                |
| 64         | High       | Fast           | Good                |
| 128        | Very High  | Fastest        | Potentially worse   |

**Commands:**
```bash
python Training.py --batch_size 16
python Training.py --batch_size 32
python Training.py --batch_size 64
python Training.py --batch_size 128
```

**Trade-offs:**
- Larger batch size → faster training (fewer updates)
- Smaller batch size → better generalization (more noise in gradients)
- GPU memory constraints

**Note:** Larger batch sizes may require higher learning rates!

---

### Recommended Experiment Order

1. **Week 1:** Baseline + 3 data augmentation experiments
2. **Week 2:** Combined augmentation + 3 architecture experiments
3. **Week 3:** Learning rate sweep (5 experiments)
4. **Week 4:** Batch size + scheduler experiments
5. **Week 5:** Final best configuration + test set evaluation

---

## 10. Code Snippets with Line Numbers

### Critical Code Sections

#### 1. Model Initialization (Training.py)

```python
# Training.py Lines 100-101
## Loading the model to run
model = timm.create_model(args.model_mode, pretrained=True, num_classes=args.n_classes)
model.to(device)
```

**Explanation:**
- `timm.create_model()`: Creates model from timm library
- `pretrained=True`: Loads ImageNet pre-trained weights
- `num_classes=543`: Replaces final layer for 543 knife classes
- Transfer learning: All layers except final are pre-trained

---

#### 2. Mixed Precision Training (Training.py)

```python
# Training.py Lines 35-43
optimizer.zero_grad()
with torch.cuda.amp.autocast():
    logits = model(images)
    loss = criterion(logits, target)

losses.update(loss.item(), images.size(0))
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Explanation:**
- `amp.autocast()`: Automatically uses FP16 where safe
- `scaler.scale()`: Prevents gradient underflow in FP16
- ~2x faster training with minimal accuracy loss

---

#### 3. mAP@5 Calculation (utils.py)

```python
# utils.py Lines 147-158
def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))
        
        # top accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)
        
        accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5
```

**Formula:**
$$\text{mAP@5} = \sum_{i=1}^{5}\frac{P(i)}{i}$$

Where $P(i)$ is the proportion of samples where correct class is in top-i predictions.

---

#### 4. Data Augmentation Pipeline (data.py)

```python
# data.py Lines 28-48
def build_transforms(self):
    if self.mode == "train":
        transform_list = [
            T.Resize((args.resized_img_weight, args.resized_img_height)),
            T.ColorJitter(brightness=args.brightness, contrast=args.contrast, 
                         saturation=args.saturation, hue=args.hue) if args.brightness > 0 or args.contrast > 0 or args.saturation > 0 or args.hue > 0 else None,
            T.RandomRotation(degrees=(0, args.random_rotation)) if args.random_rotation > 0 else None,
            T.RandomVerticalFlip(p=args.vertical_flip) if args.vertical_flip > 0 else None,
            T.RandomHorizontalFlip(p=args.horizontal_flip) if args.horizontal_flip > 0 else None,
            T.ToTensor(),
            ColorAugmentation() if args.color_aug else None,
            RandomErasing() if args.random_erase else None,
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        transform_list = [t for t in transform_list if t is not None]
        return T.Compose(transform_list)
```

**Design Pattern:**
- Conditional augmentation based on arguments
- None values filtered out
- Composed into single transform pipeline

---

#### 5. Learning Rate Scheduler (src/lr_schedulers.py)

```python
# src/lr_schedulers.py Lines 18-21
elif lr_scheduler == "multi_step":
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=stepsize, gamma=gamma
    )
```

**Example:**
- `stepsize=[20, 40]`, `gamma=0.1`, `initial_lr=0.001`
- Epochs 1-20: lr = 0.001
- Epochs 21-40: lr = 0.0001
- Epochs 41+: lr = 0.00001

---

#### 6. Checkpoint Saving (Training.py)

```python
# Training.py Lines 129-134
filename = f"Knife-{args.model_mode}-E{epoch+1}.pth"
if not os.path.exists(args.saved_checkpoint_path):
    os.mkdir(args.saved_checkpoint_path)
save_path = os.path.join(args.saved_checkpoint_path, filename)
torch.save(model.state_dict(), save_path)
```

**Output Example:**
```
Knife-Effb0/
    ├── Knife-tf_efficientnet_b0-E1.pth
    ├── Knife-tf_efficientnet_b0-E2.pth
    ├── ...
    └── Knife-tf_efficientnet_b0-E10.pth
```

---

#### 7. Logging System (Training.py)

```python
# Training.py Lines 110-123
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log.open(f"logs/log_train_{timestamp}.txt")
student_id = os.environ.get('STUDENT_ID', 'your_id')
student_name = os.environ.get('STUDENT_NAME', 'your_name')
log.write(f"Student ID:{student_id}\n")
log.write(f"Student name:{student_name}\n")
log.write(f"UUID:{uuid.uuid4()}\n")

log.write(f"==========\nArgs:{args}\n==========")

log.write("\n" + "-" * 25 + f" [START {timestamp}] " + "-" * 25 + "\n\n")
log.write('                           |  Train   |   Valid  |              |\n')
log.write(' | Mode  |  Iter  | Epoch  |   Loss   |    mAP   |     Time     |\n')
log.write('-' * 79 + '\n')
```

**Output Example:**
```
Student ID:1234567
Student name:John Doe
UUID:a1b2c3d4-e5f6-7890-abcd-ef1234567890
==========
Args:Namespace(model_mode='tf_efficientnet_b0', ...)
==========
------------------------- [START 2025-11-29_14-30-00] -------------------------

                           |  Train   |   Valid  |              |
 | Mode  |  Iter  | Epoch  |   Loss   |    mAP   |     Time     |
-------------------------------------------------------------------------------
| train | 789.0 |   0.0 |    2.456 |   0.000 |  5 min 23 sec |
| val   | 197.0 |   0.0 |    2.456 |   0.423 |  6 min 12 sec |
```

---

## 11. Execution Instructions

### Prerequisites

```bash
# Check Python version (requires 3.7+)
python --version

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Dataset Setup

```bash
# Expected structure:
# /Users/raunakburrows/
#     ├── CSWK_2025-26/          (current directory)
#     └── EEEM066_KnifeHunter/   (dataset directory)
#         ├── Train/
#         ├── Validation/
#         └── Test/
```

### Training

**Basic Training:**
```bash
# Set your student information
export STUDENT_ID="1234567"
export STUDENT_NAME="Your Name"

# Run training
python Training.py \
    --model_mode tf_efficientnet_b0 \
    --dataset_location ../EEEM066_KnifeHunter \
    --train_datacsv dataset/train.csv \
    --val_datacsv dataset/validation.csv \
    --saved_checkpoint_path Knife-Effb0 \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 0.00005 \
    --seed 42 \
    --optim adam \
    --lr-scheduler CosineAnnealingLR
```

**Using Shell Script:**
```bash
# Edit train.sh with your student info
nano train.sh  # Update STUDENT_ID and STUDENT_NAME

# Make executable and run
chmod +x train.sh
./train.sh
```

### Testing

```bash
# Set student information (if not already set)
export STUDENT_ID="1234567"
export STUDENT_NAME="Your Name"

# Run testing
python Testing.py \
    --model_mode tf_efficientnet_b0 \
    --model-path Knife-Effb0/Knife-tf_efficientnet_b0-E10.pth \
    --dataset_location ../EEEM066_KnifeHunter \
    --test_datacsv dataset/test.csv \
    --batch_size 32 \
    --seed 42 \
    --evaluate-only
```

**Using Shell Script:**
```bash
# Edit test.sh with your student info
nano test.sh

# Make executable and run
chmod +x test.sh
./test.sh
```

### Monitoring

**During Training:**
- Training progress prints to terminal in real-time
- Logs saved to `logs/log_train_YYYY-MM-DD_HH-MM-SS.txt`
- Checkpoints saved after each epoch in specified directory

**Viewing Logs:**
```bash
# View latest training log
tail -f logs/log_train_*.txt

# View all logs
ls -lt logs/
```

### Common Issues & Solutions

**1. CUDA Out of Memory**
```bash
# Solution: Reduce batch size
python Training.py --batch_size 16  # Instead of 32
```

**2. Dataset Not Found**
```bash
# Solution: Check dataset path
ls ../EEEM066_KnifeHunter/Train/  # Should show knife folders
```

**3. Module Not Found**
```bash
# Solution: Reinstall requirements
pip install -r requirements.txt --upgrade
```

---

## Summary

### Codebase Statistics

| Component | Files | Lines of Code |
|-----------|-------|---------------|
| Training & Testing | 2 | 234 |
| Data Pipeline | 1 | 78 |
| Configuration | 1 | 145 |
| Utilities | 1 | 178 |
| Optimizers & Schedulers | 2 | 116 |
| Transforms | 1 | 126 |
| **Total** | **8** | **877** |

### Key Design Principles

1. **Modularity:** Clear separation of concerns (data, training, utils)
2. **Configurability:** All hyperparameters via command-line args
3. **Reproducibility:** Seed setting for deterministic results
4. **Logging:** Comprehensive experiment tracking
5. **Efficiency:** Mixed precision training, optimized data loading

### Success Checklist for Distinction

- [ ] Run baseline with minimal augmentation
- [ ] Test at least 3 augmentation strategies
- [ ] Compare at least 3 different architectures
- [ ] Perform learning rate sweep (5+ values)
- [ ] Test batch size variations
- [ ] Document all results in tables/plots
- [ ] Analyze why techniques work/fail
- [ ] Report within 200 words per section
- [ ] Proper citations for all methods
- [ ] Reproducible with seed setting

---

**Document Version:** 1.0  
**Last Updated:** November 29, 2025  
**Generated for:** EEEM066 Coursework Assignment
