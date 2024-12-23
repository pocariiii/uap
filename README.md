# Klasifikasi Sampah dengan Deep Learning

## Deskripsi Proyek

Aplikasi ini adalah platform berbasis web yang dirancang untuk membantu pengguna mengidentifikasi jenis sampah daur ulang dengan memanfaatkan teknologi deep learning. Melalui aplikasi ini, pengguna dapat mengunggah gambar atau menggunakan kamera perangkat untuk memindai sampah yang ingin dikenali. Sistem secara otomatis akan mengklasifikasikan sampah tersebut ke dalam dua kategori utama, yaitu organik dan non organik, sehingga mendukung pengelolaan sampah yang lebih efisien dan ramah lingkungan. Klasifikasi dilakukan menjadi dua kategori utama:

- Organik
- Non Organik

Untuk dataset diambil dari [kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data).

# Distribusi Dataset

Dataset ini terdiri dari gambar yang dikategorikan ke dalam dua kelas: **Organik** dan **Non-Organik**. Berikut adalah distribusi detail dari dataset:

## Dataset Pelatihan
- **Organik (O):** 12.565 gambar
- **Non-Organik (R):** 9.999 gambar  
**Total Dataset Pelatihan:** 22.564 gambar

## Dataset Pengujian
- **Organik (O):** 1.401 gambar
- **Non-Organik (R):** 1.112 gambar  
**Total Dataset Pengujian:** 2.513 gambar

## Ringkasan
Dataset ini berisi total **25.077 gambar**, dengan **22.564 gambar** untuk pelatihan dan **2.513 gambar** untuk pengujian. Distribusi ini memastikan evaluasi model yang seimbang terhadap data yang belum pernah dilihat sebelumnya.


## Preprocessing
Pada tahap preprocessing, dataset diolah menggunakan generator untuk menyiapkan data pelatihan dan validasi. Gambar pada dataset pelatihan dan validasi dinormalisasi dengan membagi nilai piksel dengan `255` untuk mengubah skala piksel menjadi rentang [0, 1]. Dimensi gambar diubah menjadi 128x128 piksel untuk menyamakan ukuran input model. Data pelatihan dihasilkan menggunakan `ImageDataGenerator` tanpa augmentasi tambahan, sedangkan data validasi diatur tanpa pengacakan (`shuffle=False`) untuk memastikan evaluasi konsisten dengan label aslinya. Kedua generator menghasilkan batch data dengan ukuran 32 dan mendukung klasifikasi multi-kelas menggunakan `class_mode='categorical`.


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
![image](https://github.com/pocariiii/uap/blob/80657a24fe34d762dc5e19c82905c953ffc7dbfc/assets/img/train%20cnn.png)

![image](https://github.com/pocariiii/uap/blob/80657a24fe34d762dc5e19c82905c953ffc7dbfc/assets/img/loss%20cnn.png)

Dari plot histogram mengalami overvit

### Hasil Klasifikasi

Berikut adalah hasil evaluasi model menggunakan metrik **precision**, **recall**, dan **f1-score**:

| Kelas      | Precision | Recall | F1-Score | Jumlah Data |
|------------|-----------|--------|----------|-------------|
| **O**   | 0.85      | 0.96   | 0.90     | 1401         |
| **R** | 0.93      | 0.78   | 0.85     | 1112


### **Akurasi Total**
- **Accuracy**: 88%
- **Macro Avg**: 
   - Precision 0.88
   - Recall 0.87
   - F1-Score 0.88
- **Weighted Avg**: 
   - Precision 0.88
   - Recall 0.88
   - F1-Score 0.88


### 2. **MobileNetV2**

MobileNetV2 adalah arsitektur jaringan saraf konvolusional (CNN) yang dirancang untuk pengenalan gambar dan visi komputer, dengan fokus pada efisiensi dan kinerja pada perangkat dengan sumber daya terbatas, seperti smartphone. Arsitektur ini menggunakan blok residual terbalik dan konvolusi terpisah berdasarkan kedalaman untuk mengurangi jumlah parameter dan komputasi yang diperlukan.

- **Arsitektur**: MobileNetV2
- **Input**: Gambar dengan dimensi 128x128 piksel
- **Output**: Prediksi probabilitas untuk enam kelas sampah
- **Preprocessing**: Normalisasi piksel (0-1)

### Mendapatkan hasil berikut ini :
![image](https://github.com/pocariiii/uap/blob/6afc6c83339a38f211c7a805995861cc663b2cf8/assets/img/train%20mobilenetv2.png)

![image](https://github.com/pocariiii/uap/blob/6afc6c83339a38f211c7a805995861cc663b2cf8/assets/img/loss%20mobilenetv2.png)

## Classification Report

Hasil evaluasi model menunjukkan performa sebagai berikut:

| Kelas      | Precision | Recall | F1-Score | Jumlah Data |
|------------|-----------|--------|----------|-------------|
| **O**   | 0.90      | 0.96   | 0.93     | 1401         |
| **R** | 0.95      | 0.87   | 0.90     | 1112


### **Akurasi Total**
- **Accuracy**: 92%
- **Macro Avg**: 
   - Precision 0.92
   - Recall 0.91
   - F1-Score 0.92
- **Weighted Avg**: 
   - Precision 0.92
   - Recall 0.92
   - F1-Score 0.92



## Perbandingan Classification Report
- Model 1 = `CNN`
- Model 2 = `MobileNetv2`

| **Kelas**               | **Precision (MobileNetV2)** | **Recall (MobileNetV2)** | **F1-Score (MobileNetV2)** | **Jumlah Data (MobileNetV2)** | **Precision (CNN)** | **Recall (CNN)** | **F1-Score (CNN)** | **Jumlah Data (CNN)** |
|--------------------------|----------------------------|--------------------------|----------------------------|-------------------------------|---------------------|------------------|---------------------|-----------------------|
| **Organik (O)**          | 0.90                      | 0.96                     | 0.93                       | 1401                          | 0.85                | 0.96             | 0.90                | 1401                  |
| **Non-Organik (R)**      | 0.95                      | 0.87                     | 0.90                       | 1112                          | 0.93                | 0.78             | 0.85                | 1112                  |
| **Total Akurasi**        | 0.92                      | N/A                      | N/A                        | N/A                           | 0.88                | N/A              | N/A                 | N/A                   |
| **Rata-rata Macro**      | 0.92                      | 0.91                     | 0.92                       | N/A                           | 0.88                | 0.87             | 0.88                | N/A                   |
| **Rata-rata Weighted**   | 0.92                      | 0.92                     | 0.92                       | N/A                           | 0.88                | 0.88             | 0.88                | N/A                   |


Berdasarkan hasil evaluasi, model **MobileNetV2** menunjukkan kinerja yang lebih baik dibandingkan model **CNN** dalam hal akurasi total, precision, recall, dan F1-score, terutama pada kelas **Non-Organik**. MobileNetV2 mencapai akurasi total sebesar **92%**, dengan rata-rata precision dan F1-score masing-masing sebesar **0.92**, mengindikasikan konsistensi dalam mengenali kedua kelas. Sementara itu, model CNN memiliki akurasi total sebesar **88%**, dengan performa yang lebih rendah pada kelas **Non-Organik**. Dengan hasil ini, model MobileNetV2 direkomendasikan untuk digunakan dalam klasifikasi dataset sampah ini karena menawarkan performa yang lebih andal dan presisi yang lebih tinggi.

## Web 
![img](https://github.com/pocariiii/uap/blob/9afe5c5c2060ea64da62f6d2cb985bfb48661c7d/assets/img/ss.png)

![img](https://github.com/pocariiii/uap/blob/9afe5c5c2060ea64da62f6d2cb985bfb48661c7d/assets/img/ss%202.png)

![img](https://github.com/pocariiii/uap/blob/9afe5c5c2060ea64da62f6d2cb985bfb48661c7d/assets/img/ss%203.png)

# Author
202110370311421 - Angga Rofiul Putra