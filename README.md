# Cancer Cell Classification using Deep Learning

ALFI Dataset — Task 1 and Task 2

---

## Overview

This project focuses on automated cancer cell classification from time-lapse microscopy image sequences using deep learning.

The goal is to assist biological screening systems by accurately identifying different cell states, where early and reliable detection of abnormal cell behavior is critical.

Two classification tasks are addressed using the ALFI dataset:

---

# Task 1 — Binary Classification (Interphase vs Mitosis)

## Objective

The goal of Task 1 is to classify each cell sequence into one of two states:

- Interphase
- Mitosis

In biological research, detecting mitosis events is extremely important because they indicate active cell division. Missing mitosis events (false negatives) can lead to incorrect biological interpretations. Therefore, the model is designed to **maximize recall for mitosis detection**, even at the cost of slightly more false positives.

## Problem Characteristics

- Binary classification problem
- Highly imbalanced class distribution
- Temporal dependency across image sequences
- Fine-grained morphological differences between classes

---

# Task 2 — Phenotype Classification (Multi-Class)

## Objective

The goal of Task 2 is to classify cell sequences into detailed mitotic phenotypes:

- Early Mitosis  
- Late Mitosis  
- Cell Death  
- Multipolar Division  

This task is significantly more challenging because it requires distinguishing subtle visual and temporal differences between closely related biological states.

## Problem Characteristics

- Multi-class classification problem
- Severe class imbalance (especially Multipolar division)
- High similarity between early/late mitosis stages
- Rare event detection problem

---

## Dataset

This work uses the **ALFI Dataset**:

- Paper: https://www.nature.com/articles/s41597-023-02540-1  
- Dataset: https://springernature.figshare.com/articles/dataset/ALFI_dataset_final_/23798451  

### Dataset Description

- Time-lapse microscopy image sequences
- Annotated cell tracking information
- Sequence-based classification (not single images)
- Rich temporal biological dynamics

---

## Model Architecture

The model combines spatial and temporal feature learning:

### CNN Backbone
- EfficientNet-B0 pretrained on ImageNet
- Extracts spatial morphological features from each frame

### Temporal Modeling
- Bidirectional LSTM (BiLSTM)
- Captures temporal evolution of cells across frames

### Attention Mechanism
- Learns which frames in a sequence are most important
- Improves robustness to noisy or irrelevant frames

### Classification Head
- Fully connected layers
- Dropout regularization
- Batch normalization

---

## Key Features

- Sequence-based deep learning pipeline
- Spatial + temporal feature fusion
- Attention-based frame weighting
- Focal Loss for class imbalance handling
- Weighted sampling strategy
- Mixed Precision Training (AMP) for efficiency
- Gradient accumulation for large effective batch size
- Early stopping based on macro F1-score
- Group-aware splitting to prevent data leakage

---

## High Performance Computing (HPC)

Training was performed on the **Grand Valley State University (GVSU) HPC cluster**.

### Motivation

- Large-scale sequence models require significant compute
- Faster training for multiple experiments
- Enables longer sequence lengths and larger batch sizes

### SLURM Execution

```bash
sbatch train_slurm.sh
```

---

## Results

# Task 1 — Binary Classification (Interphase vs Mitosis)

### Confusion Matrix Summary

<img width="836" height="732" alt="cm_task1" src="https://github.com/user-attachments/assets/55fc5686-de5b-46ca-8bf2-200f2ef35208" />
 

### Evaluation Metrics

| Metric        | Class      | Value  |
|--------------|------------|--------|
| Precision     | Interphase | 0.8400 |
| Recall        | Interphase | 0.8400 |
| F1 Score      | Interphase | 0.8400 |
| Precision     | Mitosis    | 0.9728 |
| Recall        | Mitosis    | 0.9728 |
| F1 Score      | Mitosis    | 0.9728 |
| Accuracy      | Overall    | 0.9535 |
| F1 Macro      | Overall    | 0.9064 |
| F1 Weighted   | Overall    | 0.9535 |

### Interpretation
<img width="2147" height="596" alt="training_curves_live" src="https://github.com/user-attachments/assets/193f991a-b524-499a-ac80-3cdb29290877" />

- Strong and stable performance
- Very high mitosis detection capability (critical for medical use)
- Low false negative rate
- Suitable for screening applications

---

# Task 2 — Phenotype Classification

### Confusion Matrix

<img width="995" height="881" alt="cm_task2" src="https://github.com/user-attachments/assets/0e4b3833-913c-44f0-8d35-7a8de526f89e" />


### Evaluation Metrics

| Metric        | Class           | Value  |
|--------------|-----------------|--------|
| Precision     | Early Mitosis   | 0.9961 |
| Recall        | Early Mitosis   | 0.6530 |
| F1 Score      | Early Mitosis   | 0.7888 |
| Precision     | Late Mitosis    | 0.3131 |
| Recall        | Late Mitosis    | 0.8378 |
| F1 Score      | Late Mitosis    | 0.4559 |
| Precision     | Cell Death      | 0.8902 |
| Recall        | Cell Death      | 0.9605 |
| F1 Score      | Cell Death      | 0.9241 |
| Precision     | Multipolar      | 0.0526 |
| Recall        | Multipolar      | 0.4000 |
| F1 Score      | Multipolar      | 0.0930 |
| Accuracy      | Overall         | 0.7070 |
| F1 Macro      | Overall         | 0.5654 |
| F1 Weighted   | Overall         | 0.7712 |

### Key Observations

- **Cell Death** is well learned (F1 = 0.92)
- **Early Mitosis** shows high precision but moderate recall
- **Late Mitosis** shows high recall but low precision
- **Multipolar division** is very difficult due to extreme imbalance
<img width="2148" height="596" alt="training_curves_live" src="https://github.com/user-attachments/assets/b195029b-bd63-43c1-81d8-882f685ace94" />

### Challenges

- Severe class imbalance
- Rare class detection (especially Multipolar division)
- High similarity between mitosis stages
- Limited samples for minority classes

---

## Training Strategy

- Phase 1: Freeze CNN backbone
- Phase 2: Fine-tune full model
- Loss function: Focal Loss
- Learning rate schedule:
  - Warmup
  - Cosine decay
  - Reduce on plateau

---

## Future Work

- Improve rare class detection (oversampling / synthetic data)
- Replace BiLSTM with Transformer-based temporal models
- Develop real-time inference system
- Optimize for edge deployment
- Extend to full cell cycle classification

---

## Project Structure

```bash
.
├── checkpoints/
├── results/
├── data/
├── train.py
├── train_slurm.sh
└── README.md
```

---

## Outputs

- Training curves
- Confusion matrices
- Classification reports
- CSV logs with metrics

---

## Author

Ali Belhrak  
Master’s Student in Applied Computer Science  

---

## License

This project is intended for research and educational purposes.<img width="836" height="732" alt="cm_task1" src="https://github.com/user-attachments/assets/9f56a5f4-1db6-468d-be24-36be4b8612b1" />
