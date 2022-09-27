<center>

# Laporan Proyek Machine Learning - Muhammad Ammar Nabil

</center>

## **Project Overview**

---

Perkembangan teknologi telah meningkat dalam berbagai bidang seperti bidang ekonomi, ilmu pengetahuan, industri maupun dalam kehidupan sosial. Salah satu dampak perkembangan teknologi audio dan visual yang mana saat ini berkembang sangat pesat adalah industri Perfilman. Hingga saat ini jika di hitung jumlah judul film internasional di dunia ini sebesar 3,357,063 dan jumlah tersebut akan terus bertambah [(Sumarlin, E. et al., 2016)](https://jad.shahroodut.ac.ir/article_2390.html). Dengan adanya jumlah film yang banyak sampai saat ini, tentu hal tersebut membuat seorang user menjadi susah untuk mencari film dengan kriteria yang sama berdasarkan genre, produser, tahun dll. Maka dari itu, yang dibutuhkan oleh user adalah sebuah solusi untuk membantu dalam memilih film berdasarkan kriteria-kriteria tersebut.

<center>
<img src="https://media.istockphoto.com/photos/young-man-watching-movie-on-laptop-at-home-picture-id1353266761?k=20&m=1353266761&s=612x612&w=0&h=HdxIhOK59l-8ikoVgEbA0vltM5jyp-PzpHUJNmQjYos=" width=50%>

Gambar 1. Sistem rekomendasi merupakan salah satu solusi tersebut

</center>

Salah satu solusi dari permasalahan tersebut adalah dengan membuat sistem rekomendasi. Sampai saat ini sudah terdapat banyak aplikasi rekomendasi film yang telah beredar [(Sumarlin, E. et al., 2016)](https://jad.shahroodut.ac.ir/article_2390.html). Sistem rekomendasi dapat dibuat berdasarkan satu atau beberapa kriteria contohnya seperti sistem rekomendasi berdasarkan genre. Genre dalam film merupakan elemen utama dalam suatu sistem rekomendasi, karena dengan adanya pengklasifikasian genre dapat memudahkan sistem dalam mencari sebuah film berdasarkan tipe-tipe tertentu, penonton juga lebih mudah dalam mengidentifikasi film seperti apa yang ditayangkan [(Zhou, H. et al., 2010)](https://dl.acm.org/doi/abs/10.1145/1873951.1874068). Pada kasus ini model yang dibuat adalah sebuah model sistem rekomendasi yang digunakan untuk merekomendasikan film berdasarkan genre yang ada. Dataset yang digunakan berasal dari [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset).

## **Business Understanding**

---

### _Problem Statements_

Berdasarkan latar belakang di atas, rumusan masalah dalam kasus ini adalah :

- Bagaimana membuat model sistem rekomendasi untuk merekomendasikan film berdasarkan genre ?
- Bagaimana cara kerja model sistem rekomendasi untuk merekomendasikan film berdasarkan genre ?
- Bagaimana hasil evaluasi model yang telah dibuat untuk merekomendasikan film berdasarkan genre ?

### _Goals_

Adapun tujuan yang ingin dicapai dari kasus ini adalah sebagai berikut:

- Menghasilkan model sistem rekomendasi yang dapat digunakaan untuk merekomendasikan film berdasarkan genre
- Mengetahui cara kerja model sistem rekomendasi dalam merekomendasikan film berdasarkan genre
- Menghasilkan evaluasi model yang telah dibuat untuk merekomendasikan film berdasarkan genre

### _Solution Statements_

Adapun beberapa cara yang akan digunakan untuk menyelesaikan masalah tersebut adalah:

- Menggunakan pendekatan Content Based Filtering dengan metode vektorisasi TF-IDF dan Cosine Similarity.
- Menggunakan pendekatan Collaborative Filtering dengan metode Neural Network berdasarkan data rating user.

## **Data Understanding**

---

Dalam model ini, menggunakan dataset yang dicantumkan dalam jurnal **_"Increasing Performance of Recommender Systems by Combining Deep
Learning and Extreme Learning Machine"_** [(Nazari, Z. et al., 2022)](https://jad.shahroodut.ac.ir/article_2390.html) yang mana mengambil dataset opensource di _Kaggle_ dengan nama ["MovieLens 20M Dataset"](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset). Dataset ini berisi ±270.000 data film dengan berbagai genre. Lalu terdapat dataset rating terhadap film-film tersebut sebanyak **20.000.000 data rating**. Lalu terdapat dataset lainnya, tapi pada model ini tidak menggunakan dataset tersebut. Pada dataset tersebut masing-masing memiliki beberapa fitur yang terdiri dari

Pada file dataset **movie.csv** terdiri dari :

- `movieId` = Id masing masing movie
- `title` = Judul movie
- `genres` = Genre movie

Sedangkan pada file dataset **rating.csv** terdiri dari :

- `userId` = Id masing masing user
- `movieId` = Id masing masing movie
- `rating` = Rating user (0-5)
- `timestamp` = Genre movie

### _Exploratory Data Analysis_

#### **Variable Description**

Variable description merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data [(Dicoding, 2022)](https://dicoding.com). Salah satu contoh variable description adalah dengan menganalisa jenis tipe data masing masing feature. Dengan menganalisis tipe data peneliti dapat mengecek bagaimana sifat fitur fitur tersebut karena terkadang terdapat fitur tidak tepat. Dengan mengecek tipe data masing masing fitur peneliti akan lebih mengenal dataset tersebut.

Selain itu, melakukan penggantian tipe data agar bisa mengurangi resource saat model mentrain data yang cukup besar. Dalam model ini terdapat penggantian tipe data antara lain:

<center>

| Variable  | dtype Before Casting | dtype After Casting |
| --------- | :------------------: | :-----------------: |
| `movieId` |        int64         |      category       |
| `genres`  |        object        |      category       |
| `userId`  |        int64         |        int32        |
| `rating`  |       float64        |       float32       |

Tabel 1. Daftar Perubahan dtype dalam dataset

</center>

Dengan penggantian tipe data seperti yang ada pada Tabel 1, menjadikan komputasi lebih ringan karena masing masing variabel menjadi lebih sedikit alokasi memorinya (±600 MB ➡ ±300 MB). Hal tersebut akan berdampak pada kecepatan dan akurasi model yang dibuat

#### **Checking Null Value**

Agar model memiliki akurasi yang optimal, pengecekan nilai Null merupakan hal yang penting. Dikarenakan model akan menghitung semua data tanpa peduli nilai data tersebut apa, maka dari itu diperlukan untuk menghapus data yang kosong / Null karena model hanya bisa menghitung/memprediksi berdasarkan angka / nilai.

Pada dataset ini **tidak ditemukan data yang berisi null**. Maka dari itu tidak diperlukan tindakan untuk membuang data kosong tersebut.

## **Data Preparation**

---

Dalam kasus model ini, masing masing pendekatan (Content Based Filtering dan Collaborative Filtering) dibedakan dalam hal data preparation dikarenakan memang data yang dibutuhkan tiap tiap pendekatan berbeda. Berikut merupakan data preparatin dari masing masing pendekatan:

### _Content Based Filtering_

Dalam pendekatan Content Based Filtering dibutuhkan data yang mencakup movie tersebut (dalam hal ini adalah `movieId`) dan atribut yang akan digunakan untuk mengelompokkan movie (dalam hal ini adalah `genres`). Maka dari itu diperlukan beberapa tahap. Berikut merupakan tahapan mempersiapkan data dalam pendekatan Content Based Filtering:

1. **Menghapus film dengan genre lebih dari 1**

   Agar model lebih memahami dengan data yang dimiliki, diperlukan sebuah target yang jelas. Target dalam hal ini adalah `genres`. Dalam melakukan hal tersebut, dataset perlu dianalisa terlebih dahulu bentuk value, dan tipedatanya agar memudahkan dalam proses. Berdasarkan dataset tersebut, ditemukan bahwa dalam collumns `genre` dapat berisi 1 atau lebih genre. Untuk movie yang memiliki data lebih dari satu genre dipisahkan oleh karakter `pipe (|)`. Maka dari itu, untuk mendapatkan salah satu nilai genre dari movie tersebut dengan cara menggunakan fungsi `split('|')` dengan mengisi parameter `separator='|'`, sehingga nantinya akan mengembalikan sebuah array berupa genre. Setelah mendapatkan array tersebut, pilih `index` yang digunakan untuk nilai yang mewakili movie tersebut. Dalam kasus ini `index` pertama yang akan digunakan karena kebanyakan genre pada index pertama lebih releven dengan movie.

2. **Mengecek Null Value**

   Setelah menghapus genre yang tidak perlu, langkah selanjutnya adalah mengecek null value agar nantinya model bisa bekerja dengan optimal karena jika terdapat data null yang terhitung maka model akan terjadi error.

3. **Menghapus Data Duplikat**

   Dalam dataset data duplikat akan sangat tidak menguntungkan oleh model pada hal akurasi dikarenakan akan terjadi bias serta dalam komputasi karena data duplikat tersebut juga akan ikut terproses. Maka dari itu diperlukan pembuangan data yang duplikat.

### _Collaborative Filtering_

Dalam pendekatan Collaborative Filtering dibutuhkan data yang mencakup `rating` dari user dan atribut lainnya yang menunjang rating tersebut seperti judul film yang di rating (dalam hal ini adalah `movieId`). Untuk mendapatkan data tersebut diperlukan beberapa tahap. Berikut merupakan tahapan mempersiapkan data dalam pendekatan Collaborative Filtering:

1. **Mengencode data user dengan movie**

   Dalam pendekatan Collaborative Filtering dibutuhkan data yang mana menampung movie apa yang telah ditonton user dan movie apa yang belum ditonton movie. Maka dari itu agar mempermudah model dalam mengelompokkan movie-movie tersebut dibuatlah dictionary antara data user dengan data movie dan sebaliknya.

   Dalam proses mengencode, diperlukan sebuah data yang mana berisi movie dan user. Pada masing masing data kita akan buat dictionary masing masing dengan `key` berisi index dari movie/user tersebut dan `value` berisi nilai movie/user itu sendiri. Untuk mendapatkan index dapat menggunakan fungsi `enumerate(data, start)`. Dalam fungsi `enumerate()` akan mengembalikan 2 nilai berupa index yang berjalan dan data itu sendiri

2. **Normalisasi**

   Normalisasi dilakukan agar model dapat bekerja dengan optimal. Dengan melakukan normalisasi, dataset yang sebelumnya memiliki range yang cukup besar akan diubah menjadi range yang lebih kecil (0-1) dengan begitu komputasi model tidak akan terlalu berat.

<center>

| Descriptive Statistics |   Value    |
| :--------------------: | :--------: |
|         count          | 20000263.0 |
|          mean          |    0.7     |
|          std           |    0.2     |
|          min           |    0.0     |
|          25%           |    0.6     |
|          50%           |    0.7     |
|          75%           |    0.8     |
|          max           |    1.0     |

Tabel 2. Statistik Deskriptif Setelah Normalisasi

</center>

Seperti yang ada pada Tabel 2, nilai dari rating memiliki range antara 0-1

3. **Pembagian dataset**

   Pembagian dataset merupakan salah satu tindakan yang sangat diperlukan untuk mengetest apakah model sudah memiliki tingkat akurasi sesuai yang diharapkan atau tidak.

## **Modeling**

---

Dalam modeling, pendekatan yang digunakan terdapat 2 antara lain:

- Content Based Filtering
- Collaborative Filtering

Adapun kelebihan dan kekurangan masing masing algoritma tersebut antara lain

<center>

| No                                             | Algoritma                                                    | Kelebihan                                                               | Kekurangan |
| ---------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------- | ---------- |
| 1 <td rowspan="3">Content Based Filtering</td> | Simpel                                                       | Harus memiliki metadata yang baik                                       |
|                                                | Pilihan yang cocok untuk Unpersonalized Model Recommendation | Terkadang beberapa konten yang susah untuk dianalisis                   |
|                                                | Tidak diperlukan data dari user lain                         | Tidak mendapatkan rekomendasi berdasarkan apa yang dilakukan sebelumnya |
| 2 <td rowspan="3">Collaborative Filtering</td> | Bisa mendapatkan rekomendasi berdasarkan user yang mirip     | Komputasi yang mahal                                                    |
|                                                | Pilihan yang cocok untuk Personalized Model Recommendation   | Harus memiliki data user lainnya                                        |
|                                                | Memfasilitasi untuk berbagi pengetahuan antar user           | Rumit dalam pembuatan model                                             |

Tabel 3. Kekurangan dan Kelebihan Pendekatan Content Based, dan Collaborative Filtering

</center>

### _Content Based Filtering_

Content Based Filtering adalah salah satu pendekatan dalam sistem rekomendasi. Content Based Filtering sering digunakan untuk merekomendasikan user berdasarkan objek selain user. Dengan adanya Content Based Filtering suatu perusahaan dapat merekomendasikan produknya berdasarkan kemiripan produk dengan yang lainnya sehingga cocok untuk user yang baru melihat atau datang di situs tersebut. Dalam Content Based Filtering diperlukan metadata yang lengkap dikarenakan metadata itulah yang menjadi jantung Content Based Filtering.

Dalam membuat model sistem rekomendasi dengan pendekatan Content Based Filtering diperlukan beberapa tahap antara lain:

1. **Vektorisasi dengan metode TF-IDF**

   Dalam Content Based Filtering data target perlu di vektorisasikan agar data tersebut dapat dihitung seberapa penting data tersebut terhadap data lainnya. Dalam hal ini dikarenakan targetnya adalah `genre`, maka perlu memvektorisasikan `genre` tersebut dan melihat apakah data tersebut memiliki nilai vektor yang tinggi atau tidak

2. **Mencari Cosine Similarity**

   Setelah menghitung vektor dari masing masing `genre`, agar model mengetahui genre apakah yang saling berkaitan satu sama lain maka perlu dilakukan perhitungan cosine similarity. Dengan melakukan cosine similarity maka movie akan dapat dikelompokkan berdasarkan `genre` yang sama yang nantinya akan mengeluarkan rekomendasi movie yang memiliki genre yang sama

3. **Generate Rekomendasi**

   Setelah mendapatkan matrix dari cosine similarity maka model telah selesai dibuat. Lalu untuk menggerenate rekomendasi model hanya perlu mencari nilai yang mana cossine similaritynya tertinggi.

Berikut merupakan hasil rekomendasi dalam pendekatan Content Based Filtering

```python
# Get recommendations
film_recommendations('toy story', 15)
```

<center>

|     |                                              film | genre     |
| :-: | ------------------------------------------------: | :-------- |
|  0  |                                       The Pirates | Adventure |
|  1  |                                       Toy Story 3 | Adventure |
|  2  |                              Descent: Part 2, The | Adventure |
|  3  |                                    The Black Rose | Adventure |
|  4  |                                     Young Winston | Adventure |
|  5  |     St Trinian's 2 : The Legend of Fritton's Gold | Adventure |
|  6  |                  Sky Crawlers, The (Sukai Kurora) | Adventure |
|  7  | Shrek Forever After (A.K.A Shrek: The Final C...) | Adventure |
|  8  | Percy Jackson & The Olympians: The Lightining T.. | Adventure |
|  9  |                          B.N.B. (Bunty Aur Babli) | Adventure |
| 10  |                                             Agora | Adventure |
| 11  |                     When Dinosurs Ruled The Earth | Adventure |
| 12  |                             North Face (Nordwand) | Adventure |
| 13  |  Men Who Tread On The Tiger's Tail, The (Tora N.. | Adventure |
| 14  |                          How To Train Your Dragon | Adventure |

Tabel 4. Hasil rekomendasi 15-Top_Recommendation berdasarkan Genre

</center>

4. **Evaluasi Model**

   Untuk melihat apakah model sudah akurat atau tidak perlu melakukan penilaian matriks. Dalam kasus ini, model akan diperiksa keakuratannya dengan matriks Recommendation System Precission. Jika presisi tidak sesuai dengan yang diharapkan maka perlu adanya pengulangan pada tahap, data preparation agar data yang dihasilkan bagus.

### _Collaborative Filtering_

Collaborative Filtering adalah salah satu pendekatan dalam sistem rekomendasi. Collaborative Filtering sering digunakan untuk merekomendasikan user berdasarkan pengalaman user itu sendiri ataupun user lainnya. Dengan adanya Collaborative Filtering suatu perusahaan dapat merekomendasikan produknya berdasarkan pengalaman user sebelumnya sehingga cocok untuk user yang memiliki prefensi masing masing. Dalam Collaborative Filtering diperlukan dataset log aktivitas dari user itu sendiri.

Dalam membuat model sistem rekomendasi dengan pendekatan Collaborative Filtering diperlukan beberapa tahap antara lain:

1. **Modifikasi kelas model**

   Dalam Collaborative Filtering , model akan menggunakan sebuah kelas Neural Network model yang mana kelas tersebut nantinya akan dicustom terlebih dahulu agar sesuai dengan studi kasus saat ini. Model yang digunakan adalah RecommenderNet yang mana salah satu pretrained model yang ada dikeras dan digunakan untuk membuat sistem rekomendasi

2. **Menerapkan kelas model**

   Setelah melakukan penyesuaian terhadap kelas model, selanjutnya adalah dengan menerapkan model tersebut ke dataset yang telah dibersihkan sebelumnya. Parameter yang dipakai adalah `num_user`, `num_movie`, dan `embedding_size`

3. **Train model**

   Setelah menerapkan model tersebut ke dataset, hal yang perlu dilakukan adalah melatih model tersebut. Pada pelatihan ini model akan dilatih dengan parameter `epoch=50`, `batch_size=200.000`, dan `callbacks=EarlyStopping()`

4. **Generate Rekomendasi**

   Untuk menggenerate rekomendasi tidak hanya menggunakan fungsi `predict()` saja karena data return dari fungsi `predict()` perlu diolah terlebih dahulu agar dipahami oleh user.

Berikut merupakan hasil rekomendasi dengan pendekatan Collaborative Filtering

<center>

| Top Movie with High Ratings From User 546                            | Genre  |
| :------------------------------------------------------------------- | :----- |
| Dr. Strangelove Or: How I Learned To Stop Worrying And Love The Bomb | Comedy |
| Godfather, The                                                       | Crime  |
| William Shakespeare'S Romeo + Juliet                                 | Drama  |
| Reservoir Dogs                                                       | Crime  |
| Ice Storm, The                                                       | Drama  |
| American Beauty                                                      | Comedy |
| Ghost Dog: The Way Of The Samurai                                    | Crime  |
| City Of God (Cidade De Deus)                                         | Action |
| Lost In Translation                                                  | Comedy |
| Garden State                                                         | Comedy |

Tabel 5. 10-Top_High_Rated dari userid 546

| Top Movie Recommendation for User 546              | Genre   |
| :------------------------------------------------- | :------ |
| Vertigo                                            | Drama   |
| Rear Window                                        | Mystery |
| It Happened One Night                              | Comedy  |
| Sunset Blvd. (A.K.A. Sunset Boulevard)             | Drama   |
| 12 Angry Men                                       | Drama   |
| Best Years Of Our Lives, The                       | Drama   |
| On The Waterfront                                  | Crime   |
| 400 Blows, The (Les Quatre Cents Coups)            | Crime   |
| Rashomon (Rashômon)                                | Crime   |
| Secret In Their Eyes, The (El Secreto De Sus Ojos) | Crime   |

Tabel 6. 10-Top_Recommendation berdasarkan userid 546

</center>

5. **Evaluasi Model**

   Untuk melihat apakah model sudah akurat atau tidak perlu melakukan penilaian matriks. Dalam kasus ini, model akan diperiksa keakuratannya dengan matriks MSE (Mean Squared Error). Model akan dinilai seberapa error terhadap data benar. Jika error melebihi dengan yang diharapkan maka perlu adanya pengulangan pada tahap data preparation dan data training agar model menghasilkan rekomendasi yang tepat.

## **Evaluation**

---

Dalam menentukan model mana yang terbaik akurasinya, perlu diperhatikan terlebih dahulu metriks apa yang digunakan. Antara kasus memiliki masing masing matriks. Dalam kasus kali ini, model dibuat untuk menyelesaikan kasus sistem rekomendasi maka dari itu metriks yang digunakan antara lain

1. Recommendation System Precission
2. MSE (Mean Square Error)

- Recommendation System Precission
  Recommendation System Precission adalah salah satu matriks yang digunakan untuk mengukur seberapa presisikah sebuh model rekomendasi. Matriks tersebut didapatkan dengan menghitung jumlah rekomendasi yang benar dibagi dengan jumlah rekomendasi yang diberikan. Dalam kasus ini, Recommendation System Precission digunakan untuk menghitung model dengan pendekatan Content Based Filtering. Setelah menggunakan metriks tersebut dalam model Content Based Filtering, didapat presisi mencapai **100%** dan **recall 1.172**

<center>
<img src="https://miro.medium.com/max/1400/1*KQ0veHTnTOnBX2CeOjHaDw.png" width=50%>

Gambar 2. Rumus Recommender System Precision dan recall

</center>

- MSE (Mean Square Error)

  Semua matriks regresi pada umumnya memiliki cara kerja yang sama yaitu menghitung nilai error. Nilai errornya sendiri didapat berdasarkan pada masing masing matriks namun secara umum nilai error didapat dari nilaiBenar - nilaiPrediksi . Jika nilai prediksi semakin mendekati nilai benar maka nilai error pun akan semakin kecil (mendekati 0).

  Pada matriks MSE (Mean Square Error) nilai error didapatkan dari mengkuadratkan terlebih dahulu lalu baru di rata rata. Berikut merupakan MSE yang jika dinotasikan dalam matematika

<center>
<img src="https://www.gstatic.com/education/formulas2/472522532/en/mean_squared_error.svg" width=30%>

Gambar 3. Rumus Mean Squared Error

</center>

Dari matriks tersebut didapatkan tingkat error pada model sebagai berikut

<center>

| Epoch |  loss  | val_loss |  mse   | val_mse |
| :---: | :----: | :------: | :----: | :-----: |
|   1   | 0.6538 |  0.6472  | 0.0639 | 0.0611  |
|   2   | 1.4197 |  0.8099  | 0.1478 | 0.1398  |
|   3   | 6.1476 |  1.3699  | 0.2614 | 0.1526  |
|   4   | 6.5779 |  2.0804  | 0.1901 | 0.4447  |
|   5   | 4.5104 |  0.8934  | 0.2610 | 0.1162  |
|   6   | 4.1191 |  2.6865  | 0.1977 | 0.4805  |
|   7   | 3.3527 |  0.6211  | 0.2311 | 0.0457  |
|   8   | 4.7516 |  4.0370  | 0.2012 | 0.5024  |
|   9   | 6.0048 |  0.6586  | 0.2688 | 0.0594  |

Tabel 7. MSE Hasil Train dan Testing

<img src="https://raw.githubusercontent.com/manabil/Applied_Machine_Learning/main/assets/graph.png">

Gamber 4. Plot MSE Hasil Train dan Testing

</center>

### _Kesimpulan_

Berdasarkan tabel 7 dan precision dari masing masing pendekatan (Content Based dan Collaborative Filtering), semua pendekatan memiliki output yang cukup bagus dengan presisi rekomendasi Content Based Filtering mencapai **100 %** dan MSE (Mean Squared Error) dari Collaborative Filtering mencapai **0.2688** yang mana itu sudah cukup bagus untuk dijadikan model sistem rekomendasi. Namun jika kita bandingkan antara kedua pendekatan tersebut tentu tidak bisa dikarenakan masing masing pendekatan memiliki tujuannya sendiri sendiri.

## **Daftar Pustaka**

---

1. Dicoding, A. (2022). _Machine Learning Terapan_. Dicoding Academy. https://dicoding.com
2. Nazari, Z., Koohi, H. R., & Mousavi, J. (2022). Increasing Performance of Recommender Systems by Combining Deep Learning and Extreme Learning Machine. Journal of AI and Data Mining.
3. Sumarlin, E. W., Hansun, S., & Wiratama, Y. W. (2016). Rancang Bangun Aplikasi Rekomendasi Film dengan Menggunakan Algoritma Simple Additive Weighting. Jurnal Informatika Ahmad Dahlan, 10(2), 103958.
4. Zhou, H., Hermans, T., Karandikar, A. V., & Rehg, J. M. (2010, October). Movie genre classification via scene categorization. In Proceedings of the 18th ACM international conference on Multimedia (pp. 747-750).
