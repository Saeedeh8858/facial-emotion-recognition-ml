# Facial Emotion Recognition Models

This repository contains a comparative study and implementation of three deep learning models for facial emotion recognition using the FER2013 dataset:

- **EfficientNetV2B0**
- **Custom Convolutional Neural Network (CNN)**
- **Vision Transformer (ViT)**

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Visualization](#visualization)
- [UI Demo](#ui-demo)
- [Requirements](#requirements)
- [Usage](#usage)
- [References](#references)

---

## Overview

This project compares three state-of-the-art models for facial emotion recognition. Each model is trained and evaluated on the FER2013 dataset, with consistent preprocessing and metrics. The notebook includes code for training, evaluation, visualization, and a Gradio-based UI for live testing.

## Dataset

- **FER2013**: A public dataset containing 48x48 grayscale images labeled with 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
- Data is loaded from CSV, parsed, normalized, and split into training, validation, and test sets.

## Model Architectures

### EfficientNetV2B0
- Uses transfer learning with ImageNet weights.
- Custom layers for channel repeating and preprocessing.
- Data augmentation and class weighting for robustness.

### Custom CNN
- Sequential model with 3 convolutional blocks.
- Batch normalization, max pooling, dropout, and L2 regularization.
- Simple and fast baseline.

### Vision Transformer (ViT)
- Images split into patches and projected into embeddings.
- Multiple transformer layers with self-attention.
- Captures long-range dependencies.

## Training & Evaluation

- All models use Adam optimizer and categorical cross-entropy loss.
- Early stopping, learning rate reduction, and model checkpointing.
- Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix.

## Results

| Model                    | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
|--------------------------|---------------|----------------|--------------|----------|
| EfficientNetV2B0         | 65.6%         | 64.8%          | 63.4%        | 63.6%    |
| Custom CNN               | 61.9%         | 65.0%          | 56.0%        | 58.0%    |
| Vision Transformer (ViT) | 46.0%         | 48.0%          | 42.0%        | 44.0%    |

- EfficientNetV2B0 achieves the best overall accuracy and F1-score.
- CNN provides a strong baseline.
- ViT excels in capturing complex spatial relationships but requires more resources.

## Visualization

- Training/validation accuracy and loss curves.
- Confusion matrices for each model.
- Per-class F1-score comparisons (bar and pie charts).

## UI Demo

A Gradio web interface is provided for live emotion prediction:

- Upload a face image.
- Select a model (EfficientNetV2B0, CNN, ViT).
- Get predicted emotion.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Pandas, NumPy, Matplotlib, Seaborn, scikit-learn
- Gradio

Install dependencies:
```sh
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn gradio
```

## Usage

1. **Clone the repository**
    ```sh
    git clone https://github.com/yourusername/facial-emotion-recognition.git
    cd facial-emotion-recognition
    ```

2. **Download FER2013 dataset**
    - Place `fer2013.csv` in the appropriate directory (see notebook paths).

3. **Run the notebook**
    - Open [facialemotion.ipynb](c:\Users\Asus\Downloads\facialemotion.ipynb) in Jupyter or VS Code.
    - Execute cells to train, evaluate, and visualize models.

4. **Launch the Gradio UI**
    - Run the Gradio cell to start the web demo.

## References

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [EfficientNetV2](https://arxiv.org/abs/2104.00298)
- [Vision Transformer](https://arxiv.org/abs/2010.11929)
- [Gradio](https://gradio.app/)

---
## Additional Resources

- [Kaggle Notebook] (https://www.kaggle.com/code/saeedehalamkarue/facialemotion/edit/run/248024406)
- [Overleaf Paper] (https://www.overleaf.com/project/683dbab3c468d49a931006f0)

**Author:**  
Saeedeh Alamkar
https://github.com/Saeedeh8858
