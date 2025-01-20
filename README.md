# BACKEND QUALITY CHECKER RUPARUPA (KLASIFIKASI JENIS MEBEL)

Repository ini memaparkan project bagian backend yang memanfaatkan streamlit sebagai pengembangan website berbasis machine learning. Program ini menangani input dari user berupa gambar mebel yang nantinya dilakukan klasifikasi pada beberapa kelas seperti bed, chair, sofa, swivelchair, dan table. Program akan load model ke model klasifikasi dengan format keras(h5) bernama furniture_model.h5. Untuk hasilnya program akan menampilkan kelas terprediksi beserta akurasi dan parameter kualitas (Sangat Bagus, Bagus, Cukup Bagus, Buruk, Sangat Buruk).

## Link Deploy

[https://ruparupa.streamlit.app/]

## Library yang digunakan

- Tensorflow
- Numpy
- Pillow
- Streamlit

## Cara Menjalankan Program

Ikuti langkah-langkah di bawah ini untuk menjalankan proyek secara lokal:

```bash
# Clone repository
$ git clone https://github.com/muhzarfan/back-end-rupa2.git

# Masuk ke direktori proyek
$ cd back-end-rupa2

# Instal dependensi
$ pip install tensorflow numpy pillow streamlit

# Jalankan program
$ streamlit run app.py
```

Setelah muncul browser, anda bisa mencoba melakukan quality checker pada gambar mebel dengan mengupload suatu gambar.
