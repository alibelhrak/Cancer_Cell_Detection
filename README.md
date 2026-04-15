# Cancer Cell Detection (Interphase vs Mitosis)

## Overview

This project focuses on automated cancer cell state classification using deep learning. The goal is to detect whether a cell is in Interphase or Mitosis from microscopy image sequences.

In biological screening, missing a rare event (such as mitosis) is significantly worse than having false positives. This model prioritizes high recall for mitosis detection, making it suitable as a first-pass screening system.

---

## Dataset

This work uses the ALFI Dataset:

Paper: https://www.nature.com/articles/s41597-023-02540-1  
Dataset: https://springernature.figshare.com/articles/dataset/ALFI_dataset_final_/23798451  

### Description
- Time-lapse microscopy sequences
- Annotated cell tracks
- Two classes:
  - Mitosis
  - Interphase

---

## Model Architecture

The model combines spatial and temporal learning:

### CNN Backbone
- EfficientNet-B0 pretrained on ImageNet
- Extracts spatial features from each frame

### Temporal Modeling
- Bidirectional LSTM (BiLSTM)
- Captures temporal dependencies across sequences

### Attention Mechanism
- Learns which frames are most important

### Classification Head
- Fully connected layers with dropout and normalization

---

## Key Features

- Sequence-based classification
- Attention over time steps
- Handles class imbalance:
  - Weighted sampling
  - Focal Loss
- Mixed Precision Training (AMP)
- Gradient accumulation
- Early stopping based on macro F1-score
- Group-aware data splitting (avoids data leakage)

## Project Structure

```
.
├── checkpoints/        # Saved models
├── results/            # Metrics, plots, confusion matrices
├── data/               # ALFI dataset (not included)
├── train.py            # Main training script
└── README.md           # Project documentation
```

## Installation

pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn pillow

---

## Usage

### Train the model

python train.py \
  --data_root /path/to/ALFI \
  --epochs 35 \
  --batch_size 8

### Evaluate only

python train.py \
  --data_root /path/to/ALFI \
  --test_only \
  --checkpoint checkpoints/checkpoint_best.pt

---

## Results

### Confusion Matrix Summary

- Mitosis correctly detected: 143 (True Positives)
- Interphase correctly detected: 21 (True Negatives)
- False Positives: 4
- False Negatives: 4

---

### Evaluation Metrics

<img width="2147" height="596" alt="training_curves_live" src="https://github.com/user-attachments/assets/68ad57f6-7f41-45d8-8cd9-223f616c506d" />
<div align="center">

<table>
  <tr>
    <th>Metric</th>
    <th>Class / Scope</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Precision</td>
    <td>Interphase</td>
    <td>0.84</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>Mitosis</td>
    <td>0.9728</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>Interphase</td>
    <td>0.84</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>Mitosis</td>
    <td>0.9728</td>
  </tr>
  <tr>
    <td>F1 Score</td>
    <td>Interphase</td>
    <td>0.84</td>
  </tr>
  <tr>
    <td>F1 Score</td>
    <td>Mitosis</td>
    <td>0.9728</td>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>Overall</td>
    <td>0.9535</td>
  </tr>
  <tr>
    <td>F1 Macro</td>
    <td>Overall</td>
    <td>0.9064</td>
  </tr>
  <tr>
    <td>F1 Weighted</td>
    <td>Overall</td>
    <td>0.9535</td>
  </tr>
</table>

</div>

---

### Test Set Statistics


<div align="center">

## Training Curves

<img width="836" height="732" alt="cm_task1" src="https://github.com/user-attachments/assets/7152336e-a153-46de-8c82-e7695927645d" />


</div>

<div align="center">

<table>
  <tr>
    <th>Metric</th>
    <th>Class / Scope</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>True Positive</td>
    <td>Mitosis</td>
    <td>143</td>
  </tr>
  <tr>
    <td>True Negative</td>
    <td>Interphase</td>
    <td>21</td>
  </tr>
  <tr>
    <td>False Positive</td>
    <td>Interphase → Mitosis</td>
    <td>4</td>
  </tr>
  <tr>
    <td>False Negative</td>
    <td>Mitosis → Interphase</td>
    <td>4</td>
  </tr>
  <tr>
    <td>Test Samples</td>
    <td>Interphase</td>
    <td>25</td>
  </tr>
  <tr>
    <td>Test Samples</td>
    <td>Mitosis</td>
    <td>147</td>
  </tr>
  <tr>
    <td>Total Samples</td>
    <td>Overall</td>
    <td>172</td>
  </tr>
</table>

</div>


---

## Outputs

- Training curves
- Confusion matrix
- Classification report
- CSV file with detailed metrics

---

## Training Strategy

- Phase 1: Freeze CNN backbone
- Phase 2: Fine-tune full model
- Loss function: Focal Loss
- Scheduler:
  - Warmup
  - Cosine decay
  - Reduce on plateau

---

## Importance

Detecting mitosis is critical for cancer diagnosis and biological research. This model reduces the risk of missing important events and can assist in automated screening systems.

---

## Limitations

- Class imbalance remains a challenge
- Performance depends on sequence quality
- Requires GPU for efficient training

---

## Future Work

- Extend to full cell cycle classification
- Use transformer-based temporal models
- Real-time inference system
- Clinical deployment

---

## Author

Ali Belhrak  
Master’s Student in Applied Computer Science  

---

## License

This project is intended for research and educational purposes.
