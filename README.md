# Hybrid CNN-Transformer-GNN for Multi-Class Disaster Image Classification

Implementation of the research work:

**Cross-Attention Guided CNN-Transformer-GNN Architecture for Multi-Class Disaster Image Classification**

This repository contains the implementation and experimental reproduction of a hybrid deep learning architecture that combines a Convolutional Neural Network (CNN), Transformer Encoder, Lightweight Graph Neural Network (GNN), and Cross-Attention Gating Block (CAGB) for multi-class disaster image classification.

---

## 1. Project Overview

Natural disaster image classification is an important task for disaster monitoring, assessment, emergency response, and decision support.

The proposed framework combines:

- InceptionV3-based CNN feature extraction
- Transformer-based contextual feature learning
- Lightweight GNN-based relational feature learning
- Cross-Attention Gating Block (CAGB)
- Fully connected classification

The main objective is to jointly capture:

1. Local visual features
2. Global contextual relationships
3. Inter-regional relational information
4. Complementary information between CNN, Transformer, and GNN representations

---

## 2. Proposed Architecture

The overall architecture is:

```text
                         Input Image
                              |
              +---------------+---------------+
              |                               |
              v                               v
       InceptionV3 Branch              Transformer Branch
              |                               |
              v                               v
      CNN Feature Extraction            Stem Block
              |                               |
              v                               v
     Global Average Pooling             Patch Embedding
              |                               |
              v                               v
       CNN Projection                 Transformer Encoder
                                              |
                                              v
                                      Transformer Tokens
                                              |
                                              v
                                      Lightweight GNN
                                              |
                                              v
                                         GNN Features
              |                               |
              +---------------+---------------+
                              |
                              v
                  Cross-Attention Gating
                       Block (CAGB)
                              |
                              v
                    Feature Concatenation
                              |
                              v
                     Fused Representation
                              |
                              v
                       Dense Layer (256)
                              |
                              v
                            Softmax
                              |
                              v
                    Disaster Classification
```

The proposed framework combines CNN-based visual feature extraction, Transformer-based contextual learning, and GNN-based relational representation learning through the CAGB fusion mechanism.

---

## 3. Dataset

The paper uses the **Disaster Images Dataset (CNN Model)**.

The dataset contains **4,428 images** distributed across four disaster categories.

### 3.1 Dataset Classes

| Class | Number of Images |
|---|---:|
| Cyclone | 928 |
| Earthquake | 1,350 |
| Flood | 1,073 |
| Wildfire | 1,077 |
| **Total** | **4,428** |

### 3.2 Dataset Splitting

The dataset is divided into training, validation, and testing subsets.

| Dataset | Number of Images | Percentage |
|---|---:|---:|
| Training | 2,834 | ~64% |
| Validation | 708 | ~16% |
| Testing | 886 | ~20% |
| **Total** | **4,428** | **100%** |

The complete image collection is treated as the master image pool before splitting.

The paper also describes duplicate verification using SHA-256 file-content hashing and perceptual hashing.

The dataset itself is not stored in this GitHub repository.

---

## 4. Data Preprocessing

All images are resized to:

```text
224 × 224 pixels
```

Pixel intensities are normalized from:

```text
[0, 255]
```

to:

```text
[-1, 1]
```

The preprocessing pipeline is:

```text
Input Image
     |
     v
Resize to 224 × 224
     |
     v
Pixel Normalization
[0,255] → [-1,1]
     |
     v
Training / Validation / Testing
```

---

## 5. Data Augmentation

Online data augmentation is applied dynamically during training.

The reported augmentation configuration includes:

| Augmentation | Setting |
|---|---:|
| Rotation | 30° |
| Zoom | 0.15 |
| Horizontal shift | 0.20 |
| Vertical shift | 0.20 |
| Shear | 0.15 |
| Horizontal flip | Enabled |
| Fill mode | Nearest |
| Batch size | 16 |
| Random seed | 0 |

The augmentation is applied after dataset partitioning and is generated dynamically during training.

---

## 6. Model Architecture

The proposed model consists of four main components:

1. InceptionV3 CNN branch
2. Transformer branch
3. Lightweight Graph Neural Network
4. Cross-Attention Gating Block

These components are followed by a fully connected classification layer.

---

## 7. InceptionV3 CNN Branch

### 7.1 Purpose

The CNN branch extracts high-level visual features from disaster images.

The proposed framework uses a pretrained:

```text
InceptionV3
```

network initialized with ImageNet weights.

### 7.2 CNN Processing

The CNN branch follows:

```text
Input Image
     |
     v
InceptionV3
     |
     v
Convolutional Feature Map
     |
     v
Global Average Pooling
     |
     v
CNN Feature Vector
     |
     v
Linear Projection
     |
     v
Projected CNN Features
```

The extracted CNN representation is projected into a common feature space for subsequent fusion.

---

## 8. Transformer Branch

### 8.1 Purpose

The Transformer branch captures long-range contextual relationships between different regions of the input image.

### 8.2 Transformer Pipeline

```text
Input Image
     |
     v
Stem Block
     |
     v
Patch Embedding
     |
     v
Patch Tokens
     |
     v
Transformer Encoder
     |
     v
Context-Enriched Tokens
```

The Transformer encoder uses self-attention to model relationships between image patches.

### 8.3 Transformer Configuration

| Parameter | Value |
|---|---:|
| Patch size | 16 × 16 |
| Embedding dimension | 64 |
| Transformer blocks | 2 |
| Attention heads | 4 |
| Key dimension | 16 |
| Feed-forward layer 1 | 128 |
| Feed-forward layer 2 | 64 |

The Transformer encoder contains multi-head self-attention, residual connections, layer normalization, and feed-forward layers.

---

## 9. Lightweight Graph Neural Network

### 9.1 Purpose

The Lightweight GNN models relationships between the Transformer tokens.

The Transformer tokens are treated as graph nodes.

### 9.2 Graph Processing

The graph-learning process is:

```text
Transformer Tokens
        |
        +------------------+
        |                  |
        v                  v
 Query Projection    Key Projection
        |                  |
        +--------+---------+
                 |
                 v
          Similarity Matrix
                 |
                 v
        Attention Adjacency
                 |
                 v
        Message Aggregation
                 |
                 v
             Node Update
                 |
                 v
          GNN Features
```

The graph is dynamically constructed using learned attention relationships.

### 9.3 GNN Configuration

| Parameter | Value |
|---|---:|
| Feature dimension | 64 |
| k | 6 |

The GNN performs:

1. Query projection
2. Key projection
3. Attention computation
4. Graph construction
5. Message aggregation
6. Node updating
7. Feature projection

---

## 10. Cross-Attention Gating Block (CAGB)

### 10.1 Purpose

The **Cross-Attention Gating Block (CAGB)** adaptively fuses the CNN, Transformer, and GNN representations.

Instead of directly combining all features, the CAGB uses cross-attention and gating mechanisms to emphasize informative features and reduce redundant information.

### 10.2 Feature Interactions

The proposed feature interactions are:

```text
Transformer → CNN

CNN → Transformer

Transformer → GNN
```

### 10.3 CAGB Processing

The CNN branch is refined using Transformer-guided cross-attention:

```text
CNN Features
     |
     v
Transformer-Guided Cross-Attention
     |
     v
Gating Mechanism
     |
     v
Gated CNN Features
```

The Transformer branch is refined using CNN-guided cross-attention:

```text
Transformer Features
     |
     v
CNN-Guided Cross-Attention
     |
     v
Gating Mechanism
     |
     v
Gated Transformer Features
```

The GNN branch is refined using Transformer-guided cross-attention:

```text
GNN Features
     |
     v
Transformer-Guided Cross-Attention
     |
     v
Gating Mechanism
     |
     v
Gated GNN Features
```

The three gated representations are then concatenated:

```text
Gated CNN Features
        +
Gated Transformer Features
        +
Gated GNN Features
        |
        v
Feature Concatenation
        |
        v
Fused Feature Representation
```

### 10.4 CAGB Configuration

| Parameter | Value |
|---|---:|
| Common projection dimension | 256 |
| Attention heads | 4 |

---

## 11. Final Classification

The fused representation is passed through a fully connected layer.

```text
Fused Feature Representation
            |
            v
       Dense Layer
       256 neurons
            |
            v
           ReLU
            |
            v
         Softmax
            |
            v
     Four Disaster Classes
```

The Softmax layer produces probability scores for the four disaster categories.

The class with the maximum probability is selected as the final prediction.

---

## 12. Training Configuration

The reported implementation uses:

- Python 3.10
- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn
- GPU-enabled Google Colab

### 12.1 Training Hyperparameters

| Hyperparameter | Value |
|---|---:|
| Batch size | 16 |
| Maximum epochs | 50 |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁶ |
| Image size | 224 × 224 |
| Number of classes | 4 |

---

## 13. Evaluation Metrics

The implementation will evaluate the model using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Class-wise Classification Report

---

## 14. Reported Results

The paper reports the following final test results:

| Metric | Reported Result |
|---|---:|
| Accuracy | 96.16% |
| Macro Precision | 96.33% |
| Macro Recall | 96.10% |
| Macro F1-score | 96.20% |
| Micro ROC-AUC | 99.71% |

These values are the results reported in the paper.

They are reference values for this implementation and are **not yet claimed as independently reproduced results from this repository**.

---

## 15. Ablation Study

The architecture will be implemented progressively to evaluate the contribution of each major component.

The experimental sequence is:

```text
Experiment 1
InceptionV3
     |
     v
Experiment 2
InceptionV3 + Transformer
     |
     v
Experiment 3
InceptionV3 + Transformer + GNN
     |
     v
Experiment 4
InceptionV3 + Transformer + GNN + CAGB
```

This approach allows the contribution of the Transformer, GNN, and CAGB components to be evaluated separately.

---
