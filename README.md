# 🔬 Multi-Class Nail Pathology and Melanoma Classification via CNN & Transfer Learning

Repositori ini menyediakan pipeline Computer Vision berbasis Deep Learning menggunakan **TensorFlow / Keras** untuk mendeteksi dan mengklasifikasikan 5 kondisi abnormalitas kuku serta melanoma akral dari citra klinis.

---

## 📌 Project Overview
Visual diagnosis of nail conditions and melanoma often faces significant challenges due to severe class imbalance, where critical conditions such as melanoma contain very few clinical samples. This project tackles these challenges through:
1. **Image Verification & Cleaning:** Automated integrity verification using PIL to detect and eliminate corrupted files.
2. **Class Balancing (Oversampling & Synthetic Augmentation):** Targeted data augmentation for minority classes to achieve a balanced distribution (~650 images per class).
3. **Cost-Sensitive Learning (*Class Weights*):** Integrated balanced class weights during training to prevent minority class misclassification.
4. **Architectural Benchmarking:** Comparative evaluation among **Baseline CNN**, **Modified CNN (Batch Normalization & Dropout)**, and **Transfer Learning (MobileNetV2 pre-trained on ImageNet)**.

---

## 🏷 Pathology Classes
The models are trained to categorize images into 5 distinct classes:
1. `Acral_Lentiginous_Melanoma` (Malignant melanoma occurring on the nail bed or peripheral skin)
2. `blue_finger` (Cyanosis or bluish discoloration on fingers/nails)
3. `Healthy_Nail` (Normal/healthy nail structure)
4. `Onychogryphosis` (Severe nail hypertrophy producing a claw-like deformity)
5. `pitting` (Small punctate depressions commonly associated with psoriasis or alopecia)

---

## ⚙️ Dataset Handling & Augmentation

### Class Distribution (Before vs. After Augmentation)
| Pathology Category | Initial Count | Post-Augmentation Count |
| :--- | :---: | :---: |
| **Acral_Lentiginous_Melanoma** | 21 | **628** |
| **blue_finger** | 612 | **650** |
| **Healthy_Nail** | 343 | **648** |
| **Onychogryphosis** | 600 | **650** |
| **pitting** | 629 | **650** |

- **Augmentation Pipeline:** Rotation up to 45°, width/height shifts (0.25), shear range (0.3), zoom range (0.3), and horizontal/vertical flips.
- **Data Split:** 80% Training and 20% Validation/Testing.

---

## 🏗️ Model Architectures

1. **Baseline CNN:**
   - 3 Convolutional Blocks: `Conv2D` (32, 64, 128 filters; 3x3 kernel; ReLU activation) + `MaxPooling2D(2, 2)`.
   - Classifier: `Flatten` $\rightarrow$ `Dense(128, ReLU)` $\rightarrow$ `Dense(5, Softmax)`.
   - Optimizer: Adam, Loss: *Categorical Crossentropy*.

2. **Modified CNN (Stabilized):**
   - 3 Convolutional Blocks integrated with **`BatchNormalization()`** in each stage to stabilize gradient updates and accelerate training convergence.
   - Regularization: **`Dropout(0.3)`** layer placed before the final classification head to reduce overfitting.
   - Optimizer: Adam (*learning rate* = 0.0001).

3. **Transfer Learning (MobileNetV2):**
   - Backbone: Pre-trained `MobileNetV2` on ImageNet (`include_top=False`, weights frozen).
   - Custom Head: `GlobalAveragePooling2D` $\rightarrow$ `Dense(128, ReLU)` $\rightarrow$ `Dense(5, Softmax)`.

---

## 📈 Evaluation & Benchmark Results (Test Set: 647 Images)

| Model Architecture | Accuracy | Macro Avg Precision | Macro Avg Recall | Macro Avg F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline CNN** | 75% | 0.75 | 0.75 | 0.72 |
| **Transfer Learning (MobileNetV2)** | 90% | 0.91 | 0.90 | 0.90 |
| **Modified CNN (BatchNorm + Dropout)** | **98%** | **0.98** | **0.98** | **0.98** |

### Detailed Classification Report — Modified CNN (98% Accuracy)
```text
                            precision    recall  f1-score   support

Acral_Lentiginous_Melanoma       1.00      0.97      0.98       128
               blue_finger       0.98      1.00      0.99       129
              Healthy_Nail       0.98      0.97      0.98       130
           Onychogryphosis       0.98      0.99      0.98       130
                   pitting       0.97      0.98      0.98       130

                  accuracy                           0.98       647
                 macro avg       0.98      0.98      0.98       647
              weighted avg       0.98      0.98      0.98       647
```
