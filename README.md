# Pneumonia Detection via MobileNetV2 (Transfer Learning & FineTuning)

Deep learning system for automated pneumonia classification from chest X-ray images using a pretrained MobileNetV2 architecture.  
The pipeline follows a two-stage training strategy: feature extraction and selective fine-tuning to enhance diagnostic performance and model generalization.

---

## 📌 Project Overview

This project implements a complete medical image classification workflow:

1. Data preprocessing and normalization  
2. Transfer learning using pretrained MobileNetV2  
3. Feature extraction phase (frozen backbone)  
4. Fine-tuning phase (partial layer unfreezing)  
5. Performance monitoring across training stages  
6. ROC analysis and probabilistic evaluation  
7. Clinical-style evaluation using confusion matrix metrics  

---

## 🧠 Model Architecture

| Component | Description |
|---|---|
| Backbone | MobileNetV2 pretrained on ImageNet |
| Strategy | Transfer Learning + Fine-Tuning |
| Input Type | Chest X-Ray Images |
| Task | Binary Classification (Normal vs Pneumonia) |
| Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC |

---

## 📊 Training Performance

### Phase 1 — Feature Extraction
![Feature Extraction Results](Pneumpnia20%Detection20%via20%MobileNetV2/Feature20%Extraction20%Results.png)

| Metric | Observation |
|---|---|
| Training Loss | Decreased consistently |
| Validation Loss | Mild fluctuations with stability |
| Training Accuracy | Steady improvement |
| Validation Accuracy | Stabilized around ~0.94–0.95 |
| Training AUC | Approached ~0.994 |
| Validation AUC | ~0.98 range |

---

### Phase 2 — Fine-Tuning
![Fine Tuning Results](Pneumpnia20%Detection20%via20%MobileNetV2/Fine20%Tuning20%Results.png)

| Metric | Observation |
|---|---|
| Training Loss | Further reduction after unfreezing |
| Validation Loss | Moderate fluctuations |
| Training Accuracy | Reached ~0.96+ |
| Validation Accuracy | ~0.94–0.96 |
| Training AUC | ~0.995 |
| Validation AUC | ~0.98–0.99 |

---

## 📉 ROC Curve

![ROC Curve](Pneumpnia20%Detection20%via20%MobileNetV2/ROC20%Curve.png)

| Metric | Value |
|---|---|
| ROC-AUC | **0.992** |
| Interpretation | Excellent class separability |

---

## 🔬 Confusion Matrix

![Confusion Matrix](Pneumpnia20%Detection20%via20%MobileNetV2/Confusion20%Matrix.png)

| Actual \ Predicted | Normal | Pneumonia |
|---|---|---|
| Normal | 85 | 4 |
| Pneumonia | 9 | 142 |

---

## 📈 Final Evaluation Metrics

| Metric | Value |
|---|---|
| Accuracy | 94.58% |
| Precision (Pneumonia) | 97.26% |
| Recall / Sensitivity | 94.04% |
| Specificity | 95.51% |
| F1 Score | 95.63% |
| ROC-AUC | 0.992 |

---

## 🏥 Clinical Interpretation

- High precision → low false pneumonia alarms  
- Strong sensitivity → reliable disease detection  
- High specificity → accurate normal classification  
- Excellent ROC-AUC → strong probabilistic discrimination  

Model demonstrates strong capability for AI-assisted radiological screening and decision support.

---

## ⚙️ Technologies Used

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Model | MobileNetV2 |
| Evaluation | Scikit-learn |
| Visualization | Matplotlib / Seaborn |
| Environment | Jupyter Notebook |

---

## 🚀 How to Run

```bash
git clone <repo-url>
cd pneumonia-mobilenetv2
pip install -r requirements.txt
jupyter notebook Pneumonia_Detection_via_MobileNetV2.ipynb
```

---

## 📁 Repository Structure

```
pneumonia-mobilenetv2/
│
├── Pneumonia_Detection_via_MobileNetV2.ipynb
│
├── Pneumpnia Detection via MobileNetV2/
│   ├── Confusion Matrix.png
│   ├── Feature Extraction Results.png
│   ├── Fine Tuning Results.png
│   ├── ROC Curve.png
│   └── Threshold Tuning.png
│
└── README.md
```

---

## 📜 License

This project is for research and educational purposes.

---

## 👨‍💻 Author

**Samir Mohamed — AI & Computer Vision Engineer**
