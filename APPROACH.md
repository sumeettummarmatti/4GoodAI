# Generic Scratch Detection System - Technical Approach

## 1. Problem Understanding

**Objective:** Build a robust scratch detection system that generalizes across multiple surface types and imaging conditions.

**Key Challenge:** Creating a model that works on "generic datasets" requires handling:
- Variable scratch orientations, sizes, and intensities
- Different surface materials and textures
- Varying lighting conditions and image quality
- Class imbalance in real-world scenarios

## 2. Proposed Solution Architecture

### 2.1 Multi-Stage Pipeline Approach

```
Input Image → Preprocessing → Feature Extraction → Classification → Post-Processing
```

**Stage 1: Intelligent Preprocessing**
- Adaptive histogram equalization (CLAHE) for contrast enhancement
- Multi-scale processing (handle scratches of different sizes)
- Edge-preserving denoising (bilateral filtering)
- Data augmentation for training robustness

**Stage 2: Dual-Path Feature Extraction**
- **Path A:** Pre-trained CNN (EfficientNetV2-S or ConvNeXt-Tiny)
  - Transfer learning from ImageNet
  - Fine-tuned on scratch-specific features
- **Path B:** Traditional CV features (fallback/ensemble)
  - Gabor filters at multiple orientations
  - Local Binary Patterns (LBP) for texture
  - Edge density maps

**Stage 3: Classification Head**
- Binary classification: Scratch vs. No-Scratch
- Attention mechanism to focus on defect regions
- Confidence scoring for uncertain predictions

**Stage 4: Post-Processing**
- Grad-CAM visualization for interpretability
- Confidence thresholding with reject option
- Ensemble voting if time permits

### 2.2 Model Architecture Selection

**Primary Model: EfficientNetV2-S**

*Justification:*
- Superior accuracy/efficiency trade-off
- Proven performance on fine-grained classification
- Fast inference (~50ms on CPU)
- Compact size for deployment

**Backup Model: ConvNeXt-Tiny**
- Modern architecture with strong generalization
- Better than ViT for limited data scenarios

**Why NOT ResNet/VGG:**
- Older architectures, less parameter-efficient
- EfficientNet family consistently outperforms on defect detection tasks

## 3. Generalization Strategy

### 3.1 Aggressive Data Augmentation
```python
- Random rotation (0-360°) - scratches can be any orientation
- Random scaling (0.8-1.2x) - handle different scratch sizes
- Color jittering - simulate lighting variations
- Gaussian noise injection - handle sensor noise
- Random erasing - prevent overfitting to backgrounds
- Elastic deformations - simulate material warping
- Cutout/GridMask - improve robustness
```

### 3.2 Domain Adaptation Techniques
- **Mix-up augmentation** - blend images for smoother decision boundaries
- **Self-supervised pre-training** - learn representations from unlabeled data
- **Test-time augmentation** - average predictions over multiple augmentations

### 3.3 Cross-Dataset Validation
- Train on NEU, validate on MVTec/DAGM subsets
- Report cross-dataset performance metrics

## 4. Training Strategy

### 4.1 Loss Function
**Focal Loss + Class Balancing**
```
Loss = α * (1 - p)^γ * CE_loss
```
- Handles class imbalance naturally
- Focuses on hard examples
- α=0.75, γ=2.0 (tuned values)

### 4.2 Optimization
- Optimizer: AdamW (weight decay for regularization)
- Learning rate: 1e-4 with cosine annealing
- Batch size: 32 (balanced for 2hr training)
- Early stopping with patience=10

### 4.3 Training Phases
1. **Phase 1 (20 epochs):** Freeze backbone, train head only
2. **Phase 2 (30 epochs):** Unfreeze, fine-tune entire network
3. **Phase 3 (optional):** Knowledge distillation to smaller model

## 5. Evaluation Metrics

### 5.1 Primary Metrics
- **Precision, Recall, F1-Score** - Standard classification metrics
- **Confusion Matrix** - Detailed error analysis
- **ROC-AUC** - Threshold-independent performance

### 5.2 Generalization Metrics
- **Cross-dataset accuracy** - Test on unseen datasets
- **Robustness to perturbations** - Performance under noise/blur
- **Calibration error** - Confidence reliability

### 5.3 Practical Metrics
- **Inference time** - Real-time feasibility
- **Model size** - Deployment constraints
- **False positive rate at 95% recall** - Industrial relevance

## 6. Innovation Highlights

1. **Multi-Scale Attention Mechanism**
   - Detect scratches at multiple scales simultaneously
   - Spatial attention to focus on defect regions

2. **Uncertainty Quantification**
   - Monte Carlo Dropout for uncertainty estimates
   - Reject predictions with low confidence

3. **Explainability**
   - Grad-CAM heatmaps for each prediction
   - Feature importance analysis

4. **Lightweight Deployment**
   - ONNX export for cross-platform deployment
   - Quantization-aware training for edge devices

## 7. Expected Results

**Target Performance (on held-out test set):**
- Accuracy: ≥95%
- Precision: ≥93%
- Recall: ≥96%
- F1-Score: ≥94%
- Inference Time: <100ms per image

**Generalization (cross-dataset):**
- Accuracy drop <8% on unseen domains



## 8. Deliverables

1. **GitHub Repository:**
   - Clean, modular codebase
   - README with setup instructions
   - Requirements.txt with versions
   - Training/inference scripts

2. **HuggingFace Hub:**
   - Trained model weights
   - Model card with performance metrics
   - Interactive demo (Gradio app)

3. **Documentation:**
   - This approach document
   - Classification report
   - Confusion matrix visualization
   - Sample predictions with Grad-CAM

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Insufficient training time | Start with pre-trained weights, use small efficient model |
| Overfitting to NEU dataset | Heavy augmentation, cross-validation |
| Poor generalization | Multi-dataset validation, domain adaptation |
| Class imbalance | Focal loss, stratified sampling |
| Low-quality predictions | Confidence thresholding, ensemble methods |

## 10. Future Enhancements (Beyond Hackathon)

- Segmentation model to localize scratch regions
- Multi-class classification (scratch severity levels)
- Active learning for continuous improvement
- Mobile deployment (TFLite/CoreML)
- Real-time video processing

---

