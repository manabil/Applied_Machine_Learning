<center>

# Laporan Proyek Machine Learning - Muhammad Ammar Nabil

</center>

## **Domain Proyek**

---

Sekarang ini menentukan hunian idaman merupakan hal yang sulit. Diperlukan pemikiran yang matang dalam hal mempertimbangkan beberapa aspek. Maka dari itu diperlukan sistem pendukung keputusan untuk memecahkan masalah tersebut. Dalam pembuatan sistem pendukung keputusan tersebut sebuah model yang bagus agar bisa memprediksi dengan baik. Machine learning bisa menjadi solusi dari semua itu. Dengan machine learning, seseorang dapat membuat suatu model untuk memprediksi harga rumah. [Truong, Q. et al. (2020)](https://www.sciencedirect.com/science/article/pii/S1877050920316318) telah membuat sebuah model yang dapat digunakan untuk memprediksi harga rumah di Beijing. Dengan model tersebut akan membantu banyak ekonom perumahan untuk menentukan HPI (House Price Index) yang digunakan untuk mengukur perubahan harga perumahan di banyak negara, lalu untuk menentukan total pembayaran di muka, menentukan target pasar, perubahan tingkat default hipotek, keterjangkauan perumahan di wilayah geografis tertentu [(Truong, Q. et al., 2020)](https://www.sciencedirect.com/science/article/pii/S1877050920316318).

<center>
<img src="https://miro.medium.com/max/1024/0*YMZOAO8QE4bZ4_Rk.jpg" width=30%>

Prediksi Harga Rumah Merupakan Hal Yang Dicari Saat ini

</center>

Dikarenakan pada proses penentuan HPI dihasilkan dari perhitungan semua transaksi yang ada, membuat HPI tidak efisien untuk memprediksi harga rumah tertentu. Banyak fitur seperti distrik, usia, dan jumlah lantai juga perlu dipertimbangkan, bukan hanya penjualan berulang di dekade sebelumnya [(Truong, Q. et al., 2020)](https://www.sciencedirect.com/science/article/pii/S1877050920316318). Maka dari itu dalam memecahkan kasus ini dibutuhkan sebuah cara untuk menentukan harga rumah dengan mempertimbangkan beberapa aspek dan memiliki prediksi yang lebih akurat. Cara tersebut salah satunya dengan membuat predictive analysis. Pada kasus ini model yang dibuat adalah sebuah model regresi yang digunakan untuk menentukan harga rumah berdasarkan beberapa fitur/atribut yang ada. Dataset yang digunakan berasal dari [Housing price in Beijing](https://www.kaggle.com/datasets/ruiqurm/lianjia).

## **Business Understanding**

---

### _Problem Statements_

Berdasarkan latar belakang di atas, rumusan masalah dalam kasus ini adalah :

- Bagaimana membuat model predictive analysis untuk menentukan harga rumah ?
- Bagaimana cara kerja model predictive analysis untuk menentukan harga rumah ?
- Bagaimana hasil evaluasi model yang telah dibuat untuk menentukan harga rumah ?

### _Goals_

Adapun tujuan yang ingin dicapai dari kasus ini adalah sebagai berikut:

- Menghasilkan model predictive analysis yang dapat digunakaan untuk menentukan harga rumah
- Mengetahui cara kerja model predictive analysis dalam menentukan harga rumah
- Menghasilkan evaluasi model yang telah dibuat untuk menentukan harga rumah

### _Solution Statements_

Adapun beberapa cara yang akan digunakan untuk menyelesaikan masalah tersebut adalah:

- Menggunakan algoritma KNN dengan metriks akurasi MSE (Mean Squared Error)
- Menggunakan algoritma KNN dengan metriks akurasi RMSE (Root Mean Squared Error)
- Menggunakan algoritma Random Forest dengan metriks akurasi MSE (Mean Squared Error)
- Menggunakan algoritma Random Forest dengan metriks akurasi RMSE (Root Mean Squared Error)
- Menggunakan algoritma Boosting Algorithm dengan metriks akurasi MSE (Mean Squared Error)
- Menggunakan algoritma Boosting Algorithm dengan metriks akurasi RMSE (Root Mean Squared Error)

## **Data Understanding**

---

dataset yang berisi lebih dari 300.000 data dengan 26 variabel yang merepresentaikan harga rumah yang diperdagangkan antara tahun 2009 sampai 2018 [(Truong, Q. et al., 2020)](https://www.sciencedirect.com/science/article/pii/S1877050920316318). Variabel ini yang nantinya digunakan sebagai fitur pada dataset yang akan digunakan untuk memprediksi harga rata rata masing masing rumah per meter [(Truong, Q. et al., 2020)](https://www.sciencedirect.com/science/article/pii/S1877050920316318). Pada dataset tersebut masing-masing data terdapat 26 fitur yang terdiri dari

> - **_'url'_**: URL saat fetch data

- **_'id'_**: Id transaksi
- **_'Lng'_**: Longitudinal coordinates
- **_'Lat'_**: Latitude coordinates
- **_'Cid'_**: Id komunitas
- **_'tradeTime'_**: Waktu transaksi
- **_'DOM'_**: Waktu saat di market sebelum rumah terjual (Days on Market)
- **_'followers'_**: Jumlah orang mengikuti transaksi
- **_'totalPrice'_**: Harga total
- **_'price'_**: Harga rata-rata per meter
- **_'square'_**: Luas rumah
- **_'livingRoom'_**: Total ruang tamu
- **_'drawingRoom'_**: Total ruang gambar
- **_'kitchen'_**: Total ruang dapur
- **_'bathroom'_** Total kamar mandi
- **_'floor'_**: Total lantai
- **_'buildingType'_**: Jenis bangunan rumah
  - Tower( 1 )
  - Bungalow( 2 )
  - Combination of plate and tower( 3 )
  - Plate( 4 )
- **_'constructionTime'_**: Waktu konstruksi
- **_'renovationCondition'_**: Kondisi renovasi
  - other( 1 )
  - rough( 2 )
  - Simplicity( 3 )
  - hardcover( 4 )
- **_'buildingStructure'_**: Struktur bangunan
  - unknown( 1 )
  - mixed( 2 )
  - brick and wood( 3 )
  - brick and concrete( 4 )
  - steel( 5 )
  - steel-concrete composite ( 6 )
- **_'ladderRatio'_**: Rasio penghuni yang sama satu lantai dengan jumlah elevator/ tangga
- **_'elevator'_**: Ketersediaan elevator
  - have ( 1 )
  - not have elevator( 0 )
- **_'fiveYearsProperty'_**: Apakah rumah sudah berdiri selama 5 tahun atau tidak
  - less than 5 years( 0 )
  - more than 5 years( 1 )

### _Exploratory Data Analysis_

#### **Variable Description**

Variable description merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data [(Dicoding, 2022)](https://dicoding.com). Salah satu contoh variable description adalah dengan menganalisa jenis tipe data masing masing feature. Dengan menganalisis tipe data peneliti dapat mengecek bagaimana sifat fitur fitur tersebut karena terkadang terdapat fitur tidak tepat. Dengan mengecek tipe data masing masing fitur peneliti akan lebih mengenal dataset tersebut.

#### **Removing Null Value**

Agar model memiliki akurasi yang optimal, pengecekan nilai Null merupakan hal yang penting. Dikarenakan model akan menghitung semua data tanpa peduli nilai data tersebut apa, maka dari itu diperlukan untuk menghapus data yang kosong / Null karena model hanya bisa menghitung/memprediksi berdasarkan angka / nilai.

Pada dataset ini terdapat **lebih dari 150.000 data** yang berisi null. Maka dari itu diperlukan tindakan untuk membuang data kosong tersebut. Lalu setelah menghilangkan data Null tersebut, jumlah dataset menjadi setengahnya yaitu **± 150.000 data**

#### **Univariate Analysis**

Univariate Analysis merupakan analisis yang digunakan pada satu variabel dengan tujuan untuk mengetahui dan mengidentifikasi karakteristik dari variabel tersebut [(Yuvalianda, 2020)](https://yuvalianda.com/analisis-univariat/). Dengan melakukan univariate analisis membuat peneliti dapat melihat mengenai distribusi pada salah satu variable tersebut sehingga akan bisa mencegah adanya bias/unfair data. Dibawah ini merupakan salah satu contoh univariate analisis pada fitur buildingType

<center>
<img src="https://raw.githubusercontent.com/manabil/Applied_Machine_Learning/main/assets/univariate.png" width=30%>

Univariate Analysis pada Fitur buildingType

</center>

Dapat dilihat bahwa pada fitur buildingType distribusi datanya cenderung pada "plate" sehingga besar kemungkinan jika membuat model klasifikasi berdasarkan buildingType, maka plate akan menjadi hasil yang paling dominan. Maka dari itu peneliti harus menyamakan semua data agar distribusi datanya lebih rata

#### **Multivariate Analysis**

Multivariate Analysis adalah salah satu jenis EDA yang menunjukkan hubungan antara dua atau lebih variabel pada data [(Dicoding, 2022)](https://dicoding.com). Multivariate EDA yang menunjukkan hubungan antara dua variabel biasa disebut sebagai bivariate EDA [(Dicoding, 2022)](https://dicoding.com). Berikut merupakan adalah multivariate analysis dari dataset "House in Beijing":

<center>
<img src="https://raw.githubusercontent.com/manabil/Applied_Machine_Learning/main/assets/scatterplot.png" width=40%>

Multivariate Analysis "House in Beijing"

</center>

Dilihat dari grapik tersebut terdapat korelasi antara price dan totalPrice. Lalu terdapat kolerasi juga antara square dan price. Selain itu, bathRoom, livingRoom, drawingRoom memiliki korelasi terhadap square. Maka dari itu peneliti dapat lebih fokus menggunakan fitur fitur yang penting tersebut untuk pembuatan model kedepannya

#### **Mengatasi Outliers**

Outliers adalah nilai yang anomali dengan rata rata nilai lainnya. Dalam model, nilai outliers jika dibiarkan akan membuat model susah untuk memproses dan menghasilkan akurasi yang kurang maksimal. Maka dari itu perlu pengecekan apakah ada outliers dalam dataset, jika ada maka perlu tindakan untuk menghilangkan outliers tersebut.

Berikut merupakan grafik yang digunakan untuk memvisualisasikan data outliers pada fitur totalPrice. Titik titik disisi kanan merepresentasikan data outliers yang ada

<center>
<img src="https://raw.githubusercontent.com/manabil/Applied_Machine_Learning/main/assets/outliers.png" width=30%>

Visualisasi Outliers pada Fitur totalPrice

</center>

Dalam proses mendeteksi data outliers terdapat 3 metode yaitu:

- Hypothesis Testing
- Z-score method
- IQR Method

Pada kasus ini, metode yang digunakan adalah metode IQR. Metode IQR (Inter Quartile Range) adalah salah satu metode pendeteksian outliers dengan memanfaatkan kuartil dalam data. Cara kerjanya adalah pertama dataset akan dibagi menjadi 4 bagian dengan dibagi oleh 3 quartil. Hal tersebut dilakukan untuk mengetahui distribusi masing masing data. Dengan adanya hal tersebut data yang berada diantara kuartil pertama (Q1) sampai kuartil ketiga (Q3) pasti merupakan distribusi terbanyak pada dataset sehingga data yang berada diluar kuartil tersebut dianggap sebagai data outliers yang patut untuk dibuang. Dengan demikian interquartile range atau IQR = Q3 - Q1 [(Dicoding, 2022)](https://dicoding.com).

Dalam dataset tersebut, setelah melakukan pengecekan outliers terdapat data outliers sekitar **15.000 data outliers**. Maka dari itu diperlukan pembuangan data tersebut dari dataset.

## **Data Preparation**

---

Pada data preparation, terdapat dua metode yang digunakan yaitu

- **Pembagian dataset**

  Pembagian dataset merupakan salah satu tindakan yang sangat diperlukan untuk mengetest apakah model sudah memiliki tingkat akurasi sesuai yang diharapkan atau tidak

- **Standarisasi**

  Standarisasi digunakan untuk memudahkan model dalam memproses dataset. Dengan melakukan standarisasi , model dapat memproses lebih cepat dan lebih akurat tanpa kehilangan nilai dalam dataset. Dalam dataset ini terdapat fitur yang memiliki nilai terlalu tinggi, yaitu pada fitur price dan totalPrice. Maka dari itu sebelum membuat model perlu dilakukan standarisasi terlebih dahulu

Dalam kasus ini, **tidak ada proses Reduction Dimention** dikarenakan setelah melihat Multivariate Analysis diatas, hanya sedikit fitur yang saling berkolerasi, fitur tersebut adalah totalPrice dan price yang mana tidak mungkin untuk dijadikan satu dimensi karena totalPrice merupakan target dan price merupakan fitur. Berikut merupakan coding yang dilakukan untuk membagi dataset dan standarisasi dataset

<center>
<img src="https://raw.githubusercontent.com/manabil/Applied_Machine_Learning/main/assets/standarization.png" width=50%>

Statistik Deskriptif Setelah Standarisasi

</center>

## **Modeling**

---

Dalam modeling, algoritma yang digunakan terdapat 3 antara lain:

- K-Nearest Neighbor
- Random Forest
- Boosting Algorithm

Adapun kelebihan dan kekurangan masing masing algoritma tersebut antara lain

<center>

| No                                        | Algoritma                                     | Kelebihan                                  | Kekurangan |
| ----------------------------------------- | --------------------------------------------- | ------------------------------------------ | ---------- |
| 1 <td rowspan="3">K-Nearest Neighbor</td> | Simpel                                        | Sensitif terhadap data outliers            |
|                                           | Hampir tidak ada asumsi data                  | komputasi yang mahal (tergantung jumlah n) |
|                                           | Tidak ada re-training                         | Waktu komputasi besar(tergantung jumlah n) |
| 2 <td rowspan="3">Random Forest</td>      | Tingkat akurasi tinggi                        | Komputasi yang mahal                       |
|                                           | Tidak mudah overfit                           | Sulit untuk diintrepertasikan              |
|                                           | Tidak terlalu sensitif terhadap data outliers | Waktu komputasi besar                      |
| 3 <td rowspan="3">Boosting Algorithm</td> | Simple                                        | Sensitif terhadap data outliers            |
|                                           | Cukup bagus dalam genralisasi                 | Butuh estimator yang besar                 |
|                                           | Bagus dalam mengurangi bias dan variance      | Waktu komputasi besar                      |

Tabel Kekurangan dan Kelebihan Algoritma KNN, RF, dan AdaBoost

</center>

### _K-Nearest Neighbor_

KNN adalah salah satu algoritma yang sering digunakan dalam menyelesaikan masalah regresi dan klasifikasi. KNN sendiri bekerja dengan memperhitungkan tiap tiap data berdasarkan jarak dengan data tertentu. Pertama KNN akan menentukan secara acak data awal sebanyak n yang digunakan untuk mewakili kelas tertentu lalu setelah dipilih semua data yang mewakili n kelas, KNN akan mengecek satu satu jarak data dengan data awal tersebut, jarak yang dekat dengan kelas x akan dianggap sebagai kelas x

Dalam kasus ini, KNN digunakan untuk menyelesaikan masalah regresi. Untuk parameternya sendiri menggunakan **_n_neigbors = 10_**. Sehingga terdapat 10 data yang akan dipilih secara acak untuk mewakili tiap tiap kelas

### _Random Forest_

Random forest adalah pengembangan decision tree yang mana memilih secara acak tree yang ada sebanyak n lalu dipilih tree yang paling bagus akurasinya. Random forest juga dapat menangani masalah regresi dan klasifikasi. Pada model ini menggunakan beberapa parameter untuk mengatur kinerja Random Forest antara lain :

> - n_estimators = 50

- max_depth = 16
- random_state = 55
- n_jobs = -1

> Keterangan

- n_estimators = Banyaknya tree yang dibuat
- max_depth = Kedalaman maksimal masing masing tree
- random_state = Sebuah konstanta yang digunakan untuk menentukan pengacakan
- n_jobs = Jumlah pekerjaan yang dikerjakan dalam satu waktu

### _Boosting Algorithm_

Boosting Algorithm adalah algoritma yang terdiri dari beberapa model yang bekerja secara bersama-sama dan model dilatih secara berurutan atau dalam proses yang iteratif akan memboost atau meningkatkan algoritma sebelumnya [(Dicoding, 2022)](https://dicoding.com). Dalam Boosting Algorithm terdiri dari beberapa algoritma yang saling terkait. Adapun parameter yang digunakan adalah

> - learning_rate = 0.05

- random_state = 55

> Keterangan

- learning_rate = Sebuah konstanta yang digunakan untuk menentukan besar kecilnya proses belajar
- random_state = Sebuah konstanta yang digunakan untuk menentukan pengacakan

### _Kesimpulan_

<center>

|     | KNN     | RF       | Boosting    |
| --- | ------- | -------- | ----------- |
| MSE | 6.80438 | 0.172762 | 3379.780709 |

Tabel MSE Hasil Train

</center>

Berdasarkan matriks MSE, didapat model yang paling baik adalah **Random Forest** dengan nilai MSE terkecil mencapai **0.172762**

## **Evaluation**

---

Dalam menentukan model mana yang terbaik akurasinya, perlu diperhatikan terlebih dahulu metriks apa yang digunakan. Antara kasus regresi dan klasifikasi terdapat masing masing matriks. Dalam kasus kali ini, model dibuat untuk menyelesaikan kasus regresi maka dari itu metriks yang digunakan antara lain

> - MSE (Mean Square Error)

- RMSE (Root Mean Square Error)
- MAE (Mean Average Error)

Namun pada model ini menggunakan 2 metriks yaitu **MSE dan RMSE**. Semua matriks regresi pada umumnya memiliki cara kerja yang sama yaitu menghitung nilai error. Nilai errornya sendiri didapat berdasarkan pada masing masing matriks namun secara umum nilai error didapat dari nilaiBenar - nilaiPrediksi . Jika nilai prediksi semakin mendekati nilai benar maka nilai error pun akan semakin kecil (mendekati 0).

Pada matriks MSE (Mean Square Error) nilai error didapatkan dari mengkuadratkan terlebih dahulu lalu baru di rata rata. Berikut merupakan MSE yang jika dinotasikan dalam matematika

<center>
<img src="https://www.gstatic.com/education/formulas2/472522532/en/mean_squared_error.svg" width=30%>
</center>

Sedangkan pada matriks RMSE (Root Mean Square Error) nilai error didapat sama sepert MSE namun setelah itu diakarkan. Berikut merupakan RMSE yang jika dinotasikan dalam matematika

<center>
<img src="https://www.gstatic.com/education/formulas2/472522532/en/root_mean_square_deviation.svg" width=30%>
</center>

Dari kedua matriks tersebut didapatkan tingkat error pada masing masing algoritma sebagai berikut

<center>

|                             | Matrik   | Algoritma |  train   | test |
| --------------------------- | :------- | --------: | :------: | :--: |
| 1 <td rowspan="3">MSE</td>  | KNN      |  0.006804 | 0.009395 |
|                             | RF       |  0.000173 | 0.000573 |
|                             | Boosting |  3.379781 | 3.464695 |
| 2 <td rowspan="3">RMSE</td> | KNN      |  0.002609 | 0.003065 |
|                             | RF       |  0.000416 | 0.000757 |
|                             | Boosting |  0.058136 | 0.058862 |

Tabel MSE Hasil Train dan Testing

</center>

Jika disimpulkan maka dapat diambil kesimpulan bahwa model **Random Forest** dengan matrik MSE lah yang menjadi model terbaik karena MSE yang didapat mencapai **_0.000173_**

## **Daftar Pustaka**

---

1. Truong, Q., Nguyen, M., Dang, H., & Mei, B. (2020). Housing price prediction via improved machine learning techniques. _Procedia Computer Science_, 174, 433-442.
2. Dicoding, A. (2022). _Machine Learning Terapan_. Dicoding Academy. https://dicoding.com
3. Yuvalianda. (2020). _Analisis Univariat: Pengertian, Manfaat, Hingga Contoh Lengkap_. https://yuvalianda.com/analisis-univariat/
