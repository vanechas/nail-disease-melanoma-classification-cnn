# 🔬 Multi-Class Nail Pathology and Melanoma Classification via CNN & Transfer Learning

Repositori ini berisi implementasi *Computer Vision* berbasis *Deep Learning* menggunakan **TensorFlow / Keras** untuk mendeteksi dan mengklasifikasikan 5 kondisi abnormalitas kuku serta melanoma akral secara otomatis dari citra klinis.

---

## 📌 Ringkasan Proyek
Diagnosis visual penyakit kuku dan melanoma sering terkendala oleh keterbatasan data serta ketimpangan jumlah sampel medis (*imbalanced dataset*), di mana kasus kritis seperti melanoma memiliki jumlah sampel yang sangat sedikit. Proyek ini mengatasi tantangan tersebut melalui tahapan:
1. **Pembersihan & Verifikasi Citra:** Memeriksa integritas data menggunakan modul PIL untuk mendeteksi serta menghapus citra korup secara otomatis.
2. **Penyeimbangan Kelas (Oversampling & Augmentasi Sintetis):** Meningkatkan jumlah sampel kelas minoritas secara terarah hingga mencapai target seimbang (~650 citra per kelas).
3. **Pembobotan Biaya (*Class Weights*):** Mengintegrasikan bobot kelas seimbang (*balanced class weights*) selama pelatihan.
4. **Benchmarking Arsitektur:** Membandingkan model **Baseline CNN**, **Modified CNN (Batch Normalization & Dropout)**, dan **Transfer Learning (MobileNetV2 pra-latih ImageNet)**.

---

## 🏷️ Kategori / Kelas Patologi
Model dilatih untuk mengenali 5 kategori kondisi kuku:
1. `Acral_Lentiginous_Melanoma` (Melanoma lentiginosa akral pada kuku/kulit perifer)
2. `blue_finger` (Sianosis atau diskolorisasi kebiruan pada kuku/jari)
3. `Healthy_Nail` (Kuku sehat/normal)
4. `Onychogryphosis` (Kuku melengkung dan menebal / *ram's horn nail*)
5. `pitting` (Kuku berlubang-lubang kecil akibat psoriasis atau alopesia)

---

## ⚙️ Penanganan Dataset & Augmentasi

### Distribusi Kelas Sebelum dan Sesudah Augmentasi
| Kategori Patologi | Jumlah Awal | Jumlah Pasca-Augmentasi |
| :--- | :---: | :---: |
| **Acral_Lentiginous_Melanoma** | 21 | **628** |
| **blue_finger** | 612 | **650** |
| **Healthy_Nail** | 343 | **648** |
| **Onychogryphosis** | 600 | **650** |
| **pitting** | 629 | **650** |

- **Parameter Augmentasi:** Rotasi hingga 45°, pergeseran lebar/tinggi (*shift*) 0.25, *shear range* 0.3, pembesaran (*zoom*) 0.3, serta pembalikan (*horizontal & vertical flip*).
- **Pembagian Data:** 80% Pelatihan (*Training*), 20% Validasi & Pengujian (*Validation & Test*).

---

## 🏗️ Eksperimen & Arsitektur Model

1. **Baseline CNN:**
   - 3 Blok konvolusi: `Conv2D` (32, 64, 128 filter; kernel 3x3; aktivasi ReLU) + `MaxPooling2D(2, 2)`.
   - Klasifikator: `Flatten` $\rightarrow$ `Dense(128, ReLU)` $\rightarrow$ `Dense(5, Softmax)`.
   - Optimizer: Adam, Loss: *Categorical Crossentropy*.

2. **Modified CNN (Stabilized):**
   - 3 Blok konvolusi bertingkat yang diintegrasikan dengan **`BatchNormalization()`** di setiap blok untuk menstabilkan gradien dan mempercepat konvergensi.
   - Penambahan lapisan regulasi **`Dropout(0.3)`** sebelum *Dense layer* keluaran untuk memitigasi risiko *overfitting*.
   - Optimizer: Adam (*learning rate* = 0.0001).

3. **Transfer Learning (MobileNetV2):**
   - Base model: `MobileNetV2` pra-latih ImageNet (`include_top=False`, bobot dibekukan / *frozen*).
   - Klasifikator kustom: `GlobalAveragePooling2D` $\rightarrow$ `Dense(128, ReLU)` $\rightarrow$ `Dense(5, Softmax)`.

---

## 📈 Hasil Evaluasi & Perbandingan Model (Test Set: 647 Citra)

| Model Arsitektur | Accuracy | Macro Avg Precision | Macro Avg Recall | Macro Avg F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline CNN** | 75% | 0.75 | 0.75 | 0.72 |
| **Transfer Learning (MobileNetV2)** | 90% | 0.91 | 0.90 | 0.90 |
| **Modified CNN (BatchNorm + Dropout)** | **98%** | **0.98** | **0.98** | **0.98** |

### Laporan Klasifikasi Rinci — Modified CNN (Akurasi 98%)
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
