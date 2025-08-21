# 🦴 Bone Age Prediction using Deep Learning

<div align="center">

![Bone Age Prediction](https://img.shields.io/badge/Medical%20AI-Bone%20Age%20Prediction-blue)
![Python](https://img.shields.io/badge/Python-3.11+-brightgreen)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License](https://img.shields.io/badge/License-MIT-green)

*An AI-powered solution for automated bone age assessment from hand X-ray images*

</div>

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Technical Details](#-technical-details)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🎯 Overview

Bone age assessment is a crucial diagnostic tool in pediatric medicine for evaluating skeletal maturity and diagnosing growth disorders. This project implements state-of-the-art deep learning architectures (**DenseNet121** and **ResNet34+CBAM**) that automatically predict bone age from hand X-ray images, potentially reducing assessment time from hours to seconds while maintaining clinical accuracy.

### 🔬 Medical Significance

- **Growth Disorders**: Early detection of growth hormone deficiencies
- **Endocrine Evaluation**: Assessment of pubertal development
- **Treatment Planning**: Monitoring response to growth treatments
- **Clinical Efficiency**: Standardized, reproducible assessments

## ✨ Features

🎯 **Automated Bone Age Assessment**
- Direct age prediction from cropped and enhanced hand X-rays
- Support for pediatric patients (0–19 years)
- Clinical-grade accuracy comparable to expert radiologists

🔧 **Advanced Deep Learning Pipeline**
- DenseNet121 and ResNet34 with Convolutional Block Attention Module (CBAM)
- Auxiliary gender input integrated with visual features
- Cosine learning rate scheduling with warm-up
- AdamW optimizer with weight decay regularization
- Huber loss for robust regression

📊 **Comprehensive Analysis**
- Detailed performance metrics on training, validation, and held-out test sets
- Visualization of prediction confidence
- Error analysis and statistical evaluation
- Direct comparison of DenseNet vs. Attention-augmented ResNet

🚀 **Production Ready**
- Jupyter notebook implementation for research and development
- Modular code structure for easy integration
- Comprehensive documentation and examples

## 📊 Dataset

The model is trained on the RSNA Pediatric Bone Age Challenge dataset:

- **Training**: 12,611 images  
- **Validation**: 1,425 images  
- **Test**: 200 images (held-out for final evaluation)  
- **Image Format**: Grayscale, cropped to hand region, resized to 256×256  
- **Labels**: Bone age in months + binary gender (0=female, 1=male)  
- **Preprocessing**: MediaPipe hand detection, CLAHE contrast enhancement, grayscale conversion, normalization to [0,1]  

## 🏗️ Model Architecture

### Core Architectures
Two main models were implemented:

1. **DenseNet121**
   - Dense connectivity with 4 dense blocks (6, 12, 24, 16 layers)
   - L2 regularization and dropout
   - Gender processed via small MLP and concatenated to image features
   - Final regression head with linear output

2. **ResNet34 + CBAM**
   - Residual network with [3,4,6,3] block groups
   - Convolutional Block Attention Module for channel + spatial attention
   - Gender input fused with pooled image features
   - Deep regression head with dropout regularization

### Training Setup
- **Input Shape**: (256, 256, 1) grayscale
- **Batch Size**: 64
- **Optimizer**: AdamW with weight decay
- **Loss Function**: Huber loss (δ=15)
- **Learning Rate**: Cosine decay with warm-up (base 5e-3)
- **Early Stopping**: Based on validation MAE
- **Epochs**: Up to 60 (early stopped)

## 📈 Results

### DenseNet121
- **MAE**: 8.18 months  
- **RMSE**: 11.49 months  
- **R²**: 0.93  
- **Params**: 7.1M  

### ResNet34 + CBAM
- **MAE**: 9.77 months  
- **RMSE**: 13.01 months  
- **R²**: 0.90  
- **Params**: 22.6M  

### Model Comparison
| Model | MAE (months) | RMSE | R² | Params |
|-------|--------------|------|----|--------|
| DenseNet121 | **8.18** | 11.49 | 0.93 | 7.1M |
| ResNet34+CBAM | 9.77 | 13.01 | 0.90 | 22.6M |

➡️ **DenseNet121 consistently outperformed ResNet34+CBAM in accuracy and efficiency.**

## 🔧 Technical Details

- **Preprocessing**: MediaPipe hand cropping, CLAHE, grayscale conversion, aspect-ratio preserving resize, normalization  
- **Data Pipeline**: tf.data with caching, shuffling, batching (64), prefetching  
- **Regularization**: L2 penalties + dropout  
- **Training**: AdamW optimizer, cosine LR schedule with warm-up, Huber loss  
- **Evaluation Metrics**: MAE, RMSE, R²  

## 📁 Project Structure
*(same as original README, unchanged)*

## 🎯 Model Performance Insights

✅ **DenseNet121 is the best tradeoff**: accurate, efficient, and stable  
✅ **Attention helps** but adds complexity without consistent gains  
✅ **Robust preprocessing pipeline** improves results  

### Future Enhancements
- Integrate attention directly into DenseNet  
- Use multimodal clinical metadata (height, weight, ethnicity)  
- Explore uncertainty estimation for interpretability  
- Transfer learning from related medical imaging tasks  

## 📋 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⚠️ **MEDICAL DISCLAIMER**: This software is for research and educational purposes only. It should not be used for clinical diagnosis without validation by medical professionals.  

