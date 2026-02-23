# UBS Jewelry Image Retrieval System

Sistem pencarian perhiasan berbasis gambar (**Content-Based Image Retrieval**) menggunakan model **CLIP** yang telah di-fine-tune. Aplikasi ini dibangun menggunakan **Streamlit** untuk mempermudah pencarian produk berdasarkan kemiripan visual.

---

## Langkah-Langkah Persiapan

### 1. Download Aset dari Google Drive
Silakan akses [Link Google Drive ini](https://drive.google.com/drive/u/0/folders/1ZtSuMWQNt_ZK7sOO7BV3Pbmayo2uxD40) dan unduh file/folder berikut:

* **File:**
    * `app.py` (File utama aplikasi Streamlit)
    * `clip_database.pth` (Database fitur visual)
    * `dataset_v31_validated-20260218T082459Z-1-001.zip` (Dataset gambar)
* **Folder:**
    * `clip_jewelery_finetuned_final` (Folder bobot model hasil training)

### 2. Struktur Folder
Pastikan susunan file di VS Code Anda adalah sebagai berikut agar aplikasi berjalan tanpa error:

```
Project_UBS/
├── app.py
├── clip_database.pth
├── dataset/                    <-- Hasil ekstrak zip dataset
│   ├── Anting/
│   ├── Cincin/
│   └── ...
└── clip_jewelery_finetuned_final/ <-- Folder model dari Drive
```
___________________________
## Cara Menjalankan Aplikasi

### 1. Buka Terminal di VS Code
Buka folder proyek Anda di VS Code, lalu buka terminal baru (`Ctrl + ~` atau `Terminal > New Terminal`).

### 2. Install Library yang Diperlukan
Jalankan perintah berikut untuk menginstal semua *library* yang dibutuhkan oleh sistem:

`pip install streamlit torch transformers pillow`

### 3. Jalankan Aplikasi
Setelah instalasi selesai, jalankan perintah berikut untuk mengeksekusi program:
`streamlit run app.py`
___________________________
## Cara Penggunaan

## 1. Unggah Foto: 
Klik tombol "Unggah Foto Perhiasan" pada dashboard dan pilih gambar perhiasan dari perangkat Anda.

## 2. Atur Hasil: 
Gunakan Slider yang tersedia untuk menentukan berapa banyak jumlah hasil terbaik (Top-K) yang ingin ditampilkan.

## 3. Analisis: 
Sistem akan secara otomatis memproses gambar dan menampilkan koleksi perhiasan UBS yang paling mirip secara visual, lengkap dengan Persentase Skor Kemiripan.
