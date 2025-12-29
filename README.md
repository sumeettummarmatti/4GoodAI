# 4GoodAI - Multi-Class Surface Defect Detection

A robust AI system for detecting and classifying 6 types of surface defects on steel sheets. This project utilizes deep learning (ResNet50) to classify defects and uses Grad-CAM to visualize the exact location of the defect.

## 🚀 Features
- **Multi-Class Classification**: Detects 6 specific defect types:
  - Crazing
  - Inclusion
  - Patches
  - Pitted Surface
  - Rolled-in Scale
  - Scratches
- **Explainable AI**: Grad-CAM heatmaps highlight *where* the model sees a defect.
- **Robustness**: Handles varying lighting, textures, and scales.
- **Interactive Demo**: Built-in Gradio web app for easy testing.

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Git LFS (Large File Storage) for handling model weights.

### Setup Steps
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/sumeettummarmatti/4GoodAI.git
    cd 4GoodAI
    ```

2.  **Pull Large Files (Model Weights)**
    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Create Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Data Preparation
This project is designed to work with the **NEU-DET** surface defect dataset.

1.  **Download Data** (if not already present):
    ```bash
    python scratch-detection/scripts/download_data.py
    ```

2.  **Process and Split Data**:
    This script organizes the NEU-DET images into 6 class folders and creates stratified train/val/test splits.
    ```bash
    python scratch-detection/scripts/prepare_data_multiclass.py
    ```
    *Output location: `scratch-detection/data/processed` and `scratch-detection/data/splits`.*

## 🧠 Training
To train the model from scratch or fine-tune it on the 6 classes:

```bash
python scratch-detection/src/train.py
```
- **Config**: You can adjust hyperparameters (Epochs, Batch Size, Learning Rate) directly in `src/train.py`.
- **Default Architecture**: ResNet50 (Pre-trained on ImageNet).
- **Outputs**:
    - Best model: `scratch-detection/models/best_model.pth`
    - Training logs: `scratch-detection/results/reports/training_history.json`

## 🎮 Run Demo (Inference)
Launch the interactive web interface to test the model on your own images.

```bash
python scratch-detection/src/app.py
```
- Open the provided local URL (usually `http://127.0.0.1:7860`) in your browser.
- Upload an image to see the predicted class and attention heatmap.

## 📂 Project Structure
```
4GoodAI/
├── scratch-detection/
│   ├── data/           # Raw and processed datasets
│   ├── models/         # Saved model checkpoints (.pth)
│   ├── scripts/        # Data preparation scripts
│   │   ├── download_data.py
│   │   └── prepare_data_multiclass.py  # <-- Use this for 6-class setup
│   ├── src/            # Core source code
│   │   ├── train.py    # Training loop
│   │   ├── model.py    # Model architecture definition
│   │   └── app.py      # Gradio inference app
│   └── results/        # Training logs and reports
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 📝 Approach & Methodology
For a deep dive into the technical approach, architecture decisions, and generalization strategy, please refer to [APPROACH.md](APPROACH.md).
