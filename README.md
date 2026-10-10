# Disaster Image Classification

### Cross-Attention Guided CNN–Transformer–GNN

Hybrid deep-learning model for classifying disaster images into four categories: **Cyclone, Earthquake, Flood, and Wildfire**.

The proposed framework integrates CNN image features, Transformer token representations, graph-based feature interactions, and a Cross-Attention Gating Block (CAGB).

---

## 1. Dataset

The dataset contains 4,428 images belonging to four disaster categories.

| Class | Number of Images |
|:---|---:|
| Cyclone | 928 |
| Earthquake | 1,350 |
| Flood | 1,073 |
| Wildfire | 1,077 |
| **Total** | **4,428** |

---

## 2. Dataset Split

The data is split into training and testing subsets using `train_test_split` with `random_state=0`. A further 20% of the training portion is held out for validation using `validation_split=0.2`.

| Dataset Subset | Number of Images |
|:---|---:|
| Training | 2,834 |
| Validation | 708 |
| Testing | 886 |
| **Total** | **4,428** |

---

## 3. Data Leakage Check

Train and test file paths do not overlap. A SHA-256 comparison of file contents found no byte-identical images shared between the training and test sets.

---

## 4. Image Preprocessing

| Parameter | Value |
|:---|:---|
| Image size | 224 × 224 × 3 |
| Color mode | RGB |
| Pixel normalization | [-1, 1] |
| Random seed | 0 |

### Class Mapping

| Class | Class Index |
|:---|---:|
| Cyclone | 0 |
| Earthquake | 1 |
| Flood | 2 |
| Wildfire | 3 |

---

## 5. Data Augmentation

The notebook describes the following intended online augmentation settings for training images.

| Augmentation Parameter | Value |
|:---|---:|
| Rotation range | 30 |
| Zoom range | 0.15 |
| Width shift | 0.2 |
| Height shift | 0.2 |
| Shear range | 0.15 |
| Horizontal flip | `True` |
| Fill mode | `nearest` |

Augmented images are intended to be generated during batch preparation rather than saved as separate files. The original dataset therefore contains 4,428 images.

---

## 6. Proposed Model Architecture

### 6.1. CNN Branch

- ImageNet-pretrained InceptionV3 backbone with `include_top=False`.
- Global average pooling for image-level feature extraction.

### 6.2. Transformer Branch

- Convolutional stem with two stride-2 convolutions using 32 and 64 filters.
- 16 × 16 patch embedding.
- Token embedding dimension of 64, with 16 tokens per image.
- Two Transformer encoder blocks with four attention heads and key dimension 16.

### 6.3. Graph Neural Network Branch

- Attention-based message passing across patch tokens.
- Dense attention adjacency across all tokens, without top-k sparsification.
- Contextual relationship learning between image regions.

### 6.4. Cross-Attention Gating Block (CAGB)

- Integrates CNN, Transformer, and graph-based representations.
- Projects each representation to a common dimension of 256.
- Uses three cross-attention sigmoid gates with four attention heads.
- Passes the fused representation through Dense(256) layers and a four-class softmax classifier.

### Model Parameters

| Parameter | Count |
|:---|---:|
| Total parameters | 24,764,260 |
| Trainable parameters | 24,729,828 |
| Non-trainable parameters | 34,432 |

---

## 7. Training Configuration

| Parameter | Value |
|:---|:---|
| Optimizer | Adam |
| Learning rate | 0.000001 |
| Loss function | `categorical_crossentropy` |
| Batch size | 16 |
| Epochs run | 50 |
| Steps per epoch | 178 |
| Random seed | 0 |
| Callbacks | None |
| Final training accuracy | 99.08% |
| Final validation accuracy | 97.88% |
| Best validation accuracy | 98.02% (epoch 49) |

---

## 8. Test Results

The model was evaluated on 886 test images.

| Evaluation Metric | Value |
|:---|---:|
| Test accuracy | 96.16% |
| Macro precision | 96.33% |
| Macro recall | 96.10% |
| Macro F1-score | 96.20% |
| Micro-average ROC-AUC | 0.9971 |
| Total test images | 886 |

---

## 9. Per-Class Classification Report

| Class | Precision | Recall | F1-score | ROC-AUC | Support |
|:---|---:|---:|---:|---:|---:|
| Cyclone | 1.0000 | 0.9648 | 0.9821 | 0.9995 | 199 |
| Earthquake | 0.9504 | 0.9710 | 0.9606 | 0.9965 | 276 |
| Flood | 0.9461 | 0.9279 | 0.9369 | 0.9933 | 208 |
| Wildfire | 0.9567 | 0.9803 | 0.9684 | 0.9981 | 203 |
| **Macro average** | **0.9633** | **0.9610** | **0.9620** | — | **886** |
| **Weighted average** | **0.9620** | **0.9616** | **0.9616** | — | **886** |

---

## 10. Confusion Matrix

Rows represent actual classes, and columns represent predicted classes.

| Actual / Predicted | Cyclone | Earthquake | Flood | Wildfire |
|:---|---:|---:|---:|---:|
| Cyclone | 192 | 0 | 2 | 5 |
| Earthquake | 0 | 268 | 6 | 2 |
| Flood | 0 | 13 | 193 | 2 |
| Wildfire | 0 | 1 | 3 | 199 |

---

## 11. Software Environment

The notebook was run in Google Colab.

| Package | Version |
|:---|:---|
| Python | 3.13.16 |
| TensorFlow | 2.21.0 |
| Keras | 3.13.2 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| OpenCV | 5.0.0.93 |
| Matplotlib | 3.10.0 |

---

## 12. Repository Structure

The following structure is recommended for organizing the source code and reproducibility artifacts.

```text
disaster-cnn-transformer-gnn/
├── README.md
├── disaster_classification.ipynb
├── configs/
│   └── seed0_config.json
├── results/
│   ├── test_predictions_seed0.csv
│   ├── confusion_matrix.csv
│   └── classification_report.csv
├── checkpoints/
│   └── proposed_seed0.keras
└── environment/
    └── requirements.txt
```

*Note: This is the intended directory structure. The files should be considered publicly available only after they have been uploaded and their repository locations verified.*

---

## 13. Reproducibility and Evaluation Artifacts

The following artifacts are intended to support independent verification of the reported experimental results:

- **Source code:** `disaster_classification.ipynb`
- **Run configuration:** `seed0_config.json`
- **Per-sample predictions and ground-truth labels:** `test_predictions_seed0.csv`
- **Confusion matrix:** `confusion_matrix.csv`
- **Classification report:** `classification_report.csv`
- **Trained model checkpoint:** `proposed_seed0.keras`
- **Software environment:** `requirements.txt`

The prediction records can be used to independently recompute the evaluation metrics and confusion matrix. The configuration and model checkpoint provide additional information for examining the reported experimental run.

---
