# Multi-Class Surface Defect Detection Approach

This project implements an end-to-end deep learning pipeline to identify and categorize six distinct types of surface defects in metal materials: **Crazing, Inclusion, Patches, Pitted, Rolled-in Scale, and Scratches.**

## 1. Model Architecture
The system utilizes a custom `ScratchDetector` class that enhances a standard backbone with spatial awareness.

### 🧠 Backbone
- **ResNet-50** (Pretrained on ImageNet).
- We remove the final global pooling and classification layers to preserve spatial feature maps, allowing the model to retain location-specific information.

### 👁️ Spatial Attention Module
A custom module that helps the model "focus" on localized defect regions rather than the entire surface background.
1.  **Input**: Feature maps from the backbone.
2.  **Operation**: Computes channel-wise statistics (Average and Max pooling) followed by a $7 \times 7$ convolution.
3.  **Output**: A spatial attention map that weights the features based on relevance.

### 🎯 Classification Head
A multi-layer perceptron (MLP) designed for robust feature classification:
- **Dropout** ($p=0.3$) for regularization.
- **Linear Layer** (512 units) with ReLU activation.
- **Final Output**: Probabilities for the 6 defect classes.

---

## 2. Training Strategy
To achieve high accuracy on complex surface textures, we employed several advanced techniques:

### 📉 Loss Function
**Focal Loss** ($\gamma=2.0$, $\alpha=0.75$)
- This specifically addresses class imbalance.
- It forces the model to focus on "hard-to-classify" examples (defects that look similar to background) rather than easy ones.

### 🔄 Data Augmentation (Albumentations)
We use extensive augmentation to make the model invariant to lighting and orientation:
- **Geometric**: Random rotations (180°), elastic transformations, and optical distortions to simulate material warping.
- **Photometric**: 
  - **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement.
  - Gaussian noise injection.
  - Various blur techniques (Motion, Gaussian, Median) to simulate sensor variations.

### ⚙️ Optimization
- **Optimizer**: `AdamW` (Adam with Weight Decay) for better generalization.
- **Scheduler**: Cosine Annealing Learning Rate Scheduler to ensure smooth convergence towards the global minimum.

---

## 3. Explainability (Grad-CAM)
To ensure the model is reliable and "looking" at the actual defect, we integrated **Grad-CAM** (Gradient-weighted Class Activation Mapping).

- **Method**: The system extracts gradients from the final convolutional block of the backbone (`backbone.blocks[-1]`).
- **Visualization**: It generates a heatmap overlay on the original image, highlighting the specific areas that triggered the classification (e.g., specific scratch lines or pitted regions).

---

## 4. Performance & Results
Based on the final evaluation on the held-out test set:

- **Accuracy**: Achieved **100% Validation Accuracy** across all 6 classes.
- **Classes Evaluated**: Crazing, Inclusion, Patches, Pitted, Rolled-in Scale, Scratches.
- **Hardware**: optimized for efficient CPU/GPU inference.

---

## 5. User Interface
The model is deployed via a user-friendly **Gradio** web interface in `src/app.py`.

- **Input**: Users upload a raw surface image.
- **Output**: Real-time prediction of defect category and confidence level.
- **Explainability**: Interactive heatmap showing exactly where the defect was detected.