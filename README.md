# Klasifikasi Sampah dengan Deep Learning

## Deskripsi Proyek

Aplikasi ini adalah platform berbasis web yang dirancang untuk membantu pengguna mengidentifikasi jenis sampah daur ulang dengan memanfaatkan teknologi deep learning. Melalui aplikasi ini, pengguna dapat mengunggah gambar atau menggunakan kamera perangkat untuk memindai sampah yang ingin dikenali. Sistem secara otomatis akan mengklasifikasikan sampah tersebut ke dalam enam kategori utama, yaitu Kaca, Kardus, Kertas, Logam, Plastik, dan Residu, sehingga mendukung pengelolaan sampah yang lebih efisien dan ramah lingkungan. Klasifikasi dilakukan menjadi enam kategori utama:

- Kaca
- Kardus
- Kertas
- Logam
- Plastik
- Residu

Untuk dataset sampah diambil dari [kaggle](https://www.kaggle.com/datasets/fathurrahmanalfarizy/sampah-daur-ulang).

## Augmentasi
Dataset awal terdiri dari 7014 gambar dengan distribusi per kelas sebagai berikut: Kaca (1110 gambar), Kardus (624 gambar), Kertas (1807 gambar), Plastik (1257 gambar), Residu (1006 gambar), dan Logam (1210 gambar). Untuk meningkatkan jumlah data dan memperbaiki keseimbangan antar kelas, dilakukan teknik augmentasi data sehingga jumlah gambar per kelas menjadi: Kaca (1800 gambar), Kardus (1248 gambar), Kertas (1807 gambar), Plastik (1800 gambar), Residu (1800 gambar), dan Logam (1800 gambar). Setelah augmentasi, total dataset meningkat menjadi 10.255 gambar.

## Preprocessing
Pada tahap preprocessing, dataset diolah menggunakan generator untuk menyiapkan data pelatihan dan validasi. Gambar pada dataset pelatihan dan validasi dinormalisasi dengan membagi nilai piksel dengan `255` untuk mengubah skala piksel menjadi rentang [0, 1]. Dimensi gambar diubah menjadi 128x128 piksel untuk menyamakan ukuran input model. Data pelatihan dihasilkan menggunakan `ImageDataGenerator` tanpa augmentasi tambahan, sedangkan data validasi diatur tanpa pengacakan (`shuffle=False`) untuk memastikan evaluasi konsisten dengan label aslinya. Kedua generator menghasilkan batch data dengan ukuran 32 dan mendukung klasifikasi multi-kelas menggunakan `class_mode='categorical`.

### Tujuan Pengembangan

1. Meningkatkan kesadaran tentang pengelompokan sampah untuk mendukung daur ulang.
2. Memberikan alat berbasis AI yang mudah digunakan untuk mengenali jenis sampah.
3. Mengintegrasikan teknologi deep learning dengan antarmuka yang ramah pengguna.

---

## Langkah Instalasi

Ikuti langkah-langkah berikut untuk menginstal dependencies dan menjalankan aplikasi web:

### 1. Clone Repository

Clone repositori ini ke komputer Anda:

```bash
git clone https://github.com/pocariiii/uap.git
cd uap
```

### 2. Instal Dependencies

Instal semua dependencies yang diperlukan dengan perintah berikut:

```bash
pip install -r requirements.txt
```
di dalam file `requirements.txt` berisi library dibawah ini
```
streamlit
tensorflow
pandas
pillow
altair
```

### 3. Jalankan Aplikasi

Jalankan aplikasi menggunakan perintah berikut:

```bash
streamlit run src/klasifikasi/app.py
```

### 4. Akses Aplikasi Web

Buka browser Anda dan akses aplikasi melalui URL berikut:

```
http://localhost:8501
```

---

## Deskripsi Model

Proyek ini menggunakan dua model deep learning untuk klasifikasi sampah:

### 1. **Convolutional Neural Network (CNN)**

Convolutional Neural Network (CNN) adalah arsitektur deep learning yang dirancang untuk pemrosesan data grid, seperti gambar. CNN bekerja dengan menggunakan lapisan konvolusi untuk mengekstraksi fitur dari gambar input, diikuti dengan pooling untuk mengurangi dimensi data tanpa kehilangan informasi penting.

klasifikasi gambar dengan arsitektur sebagai berikut:

| Layer (type)                 | Output Shape        | Param #      |
|------------------------------|---------------------|--------------|
| **conv2d_18 (Conv2D)**       | (None, 126, 126, 64) | 1,792        |
| **max_pooling2d_18 (MaxPooling2D)** | (None, 63, 63, 64)  | 0            |
| **conv2d_19 (Conv2D)**       | (None, 61, 61, 128)  | 73,856       |
| **max_pooling2d_19 (MaxPooling2D)** | (None, 30, 30, 128) | 0            |
| **conv2d_20 (Conv2D)**       | (None, 28, 28, 256)  | 295,168      |
| **max_pooling2d_20 (MaxPooling2D)** | (None, 14, 14, 256) | 0            |
| **flatten_6 (Flatten)**      | (None, 50176)        | 0            |
| **dense_12 (Dense)**         | (None, 512)          | 25,690,624   |
| **dropout_10 (Dropout)**     | (None, 512)          | 0            |
| **dense_13 (Dense)**         | (None, 6)            | 3,078        |

### Total Parameter:
- **Total params**: 26,064,518 (99.43 MB)
- **Trainable Params**: 26,064,518 (99.43 MB)
- **Non-trainable Params**: 0 (0.00 B)

### Mendapatkan hasil sebagai berikut : 
![image](https://github.com/pocariiii/uap/blob/a8496a6ce72998e73defaa796ae12cd63266e07a/assets/img/epoch_cnn_akurasi.png)

![image](https://github.com/pocariiii/uap/blob/dc1ee231c7dea72fedf00ef2c3246ef0927f582c/assets/img/epoch%20cnn%20loss.png)

Dari plot histogram mengalami overvit

### Hasil Klasifikasi

Berikut adalah hasil evaluasi model menggunakan metrik **precision**, **recall**, dan **f1-score**:

| Kelas      | Precision | Recall | F1-Score | Jumlah Data |
|------------|-----------|--------|----------|-------------|
| **Kaca**   | 0.72      | 0.73   | 0.72     | 540         |
| **Kardus** | 0.73      | 0.75   | 0.74     | 375         |
| **Kertas** | 0.74      | 0.65   | 0.69     | 543         |
| **Logam**  | 0.64      | 0.69   | 0.66     | 540         |
| **Plastik**| 0.66      | 0.69   | 0.67     | 540         |
| **Residu** | 0.69      | 0.69   | 0.69     | 540         |

### **Akurasi Total**
- **Accuracy**: 69%
- **Macro Avg**: 
   - Precision 0.70
   - Recall 0.70
   - F1-Score 0.70
- **Weighted Avg**: 
   - Precision 0.70
   - Recall 0.69
   - F1-Score 0.69


### 2. **MobileNetV2**

MobileNetV2 adalah arsitektur jaringan saraf konvolusional (CNN) yang dirancang untuk pengenalan gambar dan visi komputer, dengan fokus pada efisiensi dan kinerja pada perangkat dengan sumber daya terbatas, seperti smartphone. Arsitektur ini menggunakan blok residual terbalik dan konvolusi terpisah berdasarkan kedalaman untuk mengurangi jumlah parameter dan komputasi yang diperlukan.

- **Arsitektur**: MobileNetV2
- **Input**: Gambar dengan dimensi 128x128 piksel
- **Output**: Prediksi probabilitas untuk enam kelas sampah
- **Preprocessing**: Normalisasi piksel (0-1)

### Mendapatkan hasil berikut ini :
![image](https://github.com/pocariiii/uap/blob/b33d8a1e9085566df87f3a7e12786c47bf4c8115/assets/img/akurasi%20mobilenetv2.png)

![image](https://github.com/pocariiii/uap/blob/b33d8a1e9085566df87f3a7e12786c47bf4c8115/assets/img/loss%20mobilenetv2.png)

## Classification Report

Hasil evaluasi model menunjukkan performa sebagai berikut:

| Kelas      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| **Kaca**   | 0.91      | 0.88   | 0.89     | 535     |
| **Kardus** | 0.87      | 0.92   | 0.90     | 369     |
| **Kertas** | 0.90      | 0.88   | 0.89     | 543     |
| **Logam**  | 0.87      | 0.89   | 0.88     | 536     |
| **Plastik**| 0.85      | 0.82   | 0.84     | 538     |
| **Residu** | 0.85      | 0.85   | 0.85     | 530     |

### **Akurasi Total**
- **Accuracy**: 87%
- **Macro Average**:
  - Precision: 0.87
  - Recall: 0.88
  - F1-Score: 0.87
- **Weighted Average**:
  - Precision: 0.87
  - Recall: 0.87
  - F1-Score: 0.87



## Perbandingan Classification Report
- Model 1 = `CNN`
- Model 2 = `MobileNetv2`

| Kelas      | Precision (Model 1) | Recall (Model 1) | F1-Score (Model 1) | Precision (Model 2) | Recall (Model 2) | F1-Score (Model 2) |
|------------|----------------------|------------------|--------------------|----------------------|------------------|--------------------|
| **Kaca**   | 0.72                 | 0.73             | 0.72               | 0.91                 | 0.88             | 0.89               |
| **Kardus** | 0.73                 | 0.75             | 0.74               | 0.87                 | 0.92             | 0.90               |
| **Kertas** | 0.74                 | 0.65             | 0.69               | 0.90                 | 0.88             | 0.89               |
| **Logam**  | 0.64                 | 0.69             | 0.66               | 0.87                 | 0.89             | 0.88               |
| **Plastik**| 0.66                 | 0.69             | 0.67               | 0.85                 | 0.82             | 0.84               |
| **Residu** | 0.69                 | 0.69             | 0.69               | 0.85                 | 0.85             | 0.85               |

### **Akurasi Total**
| Metric              | Model 1 | Model 2 |
|---------------------|---------|---------|
| **Accuracy**        | 69%     | 87%     |
| **Macro Avg**       | 0.70    | 0.87    |
| **Weighted Avg**    | 0.70    | 0.87    |

Kesimpulannya, model ``MobileNetV2`` menunjukkan kinerja yang jauh lebih unggul dibandingkan model ``CNN``, dengan akurasi total sebesar 87% dibandingkan 69%. MobileNetV2 menunjukkan peningkatan signifikan pada semua metrik, termasuk precision, recall, dan F1-score di seluruh kelas, terutama pada kelas ``Kaca`` dan ``Kardus``, yang mengalami peningkatan F1-score dari 0.72 menjadi 0.89 dan dari 0.74 menjadi 0.90. Dengan performa yang lebih andal dan akurat, ``MobileNetV2`` direkomendasikan untuk implementasi klasifikasi sampah. Sementara itu, model CNN dapat digunakan sebagai baseline, namun memerlukan optimasi lebih lanjut untuk meningkatkan performanya.

## Web 
![img](https://github.com/pocariiii/uap/blob/f1fdd6e73d309595f287e19b1d07699355d4dcc7/assets/img/awal.png)

![img](https://github.com/pocariiii/uap/blob/f1fdd6e73d309595f287e19b1d07699355d4dcc7/assets/img/hasilweb.png)

![img](https://github.com/pocariiii/uap/blob/f1fdd6e73d309595f287e19b1d07699355d4dcc7/assets/img/akurasinya.png)
