# Healthcare Analysis dengan PySpark

Analisis data pasien rumah sakit menggunakan **Apache Spark (PySpark)**. Proyek ini adalah tugas presentasi mata kuliah Data Mining (`Presentasi_DATMIN.ipynb`). Cakupannya mulai dari pembersihan data, agregasi, klasifikasi, regresi, klastering, sampai penambangan pola asosiasi, semuanya diproses dengan PySpark.

## Daftar Isi

- [Tujuan](#tujuan)
- [Dataset](#dataset)
- [Teknologi](#teknologi)
- [Alur Analisis](#alur-analisis)
- [Ringkasan Hasil](#ringkasan-hasil)
- [Struktur Proyek](#struktur-proyek)
- [Cara Menjalankan](#cara-menjalankan)
- [Catatan dan Keterbatasan](#catatan-dan-keterbatasan)

## Tujuan

1. Memperkenalkan penggunaan PySpark untuk memproses dan menganalisis data dalam skala besar.
2. Membandingkan performa Spark dengan jumlah core yang berbeda (`local[1]` vs `local[2]`).
3. Menerapkan beberapa teknik data mining pada data kesehatan pasien:
   - Klasifikasi hasil tes (`Test Results`)
   - Regresi jumlah tagihan (`Billing Amount`)
   - Klastering pasien berdasarkan usia
   - Frequent pattern mining antara kondisi medis, jenis rawat inap, dan obat

## Dataset

File: `healthcare_rows.csv`, berisi **55.500 baris** dan **15 kolom**. Di notebook, data yang sama dibaca dari tabel `healthcare` di database PostgreSQL (Supabase) lewat `psycopg2`, lalu diubah menjadi Spark DataFrame.

| Kolom | Tipe | Keterangan |
|---|---|---|
| `Name` | String | Nama pasien (identitas, tidak dipakai sebagai fitur) |
| `Age` | Integer | Usia pasien saat dirawat |
| `Gender` | Kategorikal | Jenis kelamin (Male/Female) |
| `Blood Type` | Kategorikal | Golongan darah |
| `Medical Condition` | Kategorikal | Diagnosis utama (Arthritis, Asthma, Cancer, Diabetes, Hypertension, Obesity) |
| `Date of Admission` | String (tanggal) | Tanggal masuk rumah sakit |
| `Doctor` | String | Dokter yang menangani |
| `Hospital` | Kategorikal | Rumah sakit tempat dirawat |
| `Insurance Provider` | Kategorikal | Penyedia asuransi (Aetna, Blue Cross, Cigna, Medicare, UnitedHealthcare) |
| `Billing Amount` | Float | Total biaya perawatan |
| `Room Number` | Integer | Nomor kamar |
| `Admission Type` | Kategorikal | Emergency / Elective / Urgent |
| `Discharge Date` | String (tanggal) | Tanggal keluar |
| `Medication` | Kategorikal | Obat yang diberikan (Aspirin, Ibuprofen, Lipitor, Paracetamol, Penicillin) |
| `Test Results` | Kategorikal | Normal / Abnormal / Inconclusive |

## Teknologi

- Python 3
- PySpark (`pyspark.sql`, `pyspark.ml`)
- pandas dan psycopg2 (mengambil data dari PostgreSQL)
- Jupyter Notebook / Google Colab

## Alur Analisis

### 1. Pemuatan data dan SparkSession

Data diambil dari PostgreSQL ke pandas, lalu dikonversi ke Spark DataFrame:

```python
spark = SparkSession.builder.appName("Healthcare Analysis").master("local[2]").getOrCreate()
spark_df = spark.createDataFrame(df)
```

Jumlah partition DataFrame mengikuti nilai `master`, dan `sc.defaultParallelism` menampilkan tingkat paralelismenya.

**Uji perbedaan kecepatan.** Sel yang sama dijalankan dengan dua konfigurasi:

| Konfigurasi | Partition | Waktu eksekusi sel |
|---|---|---|
| `local[1]` | 1 | ± 20 detik |
| `local[2]` | 2 | ± 15 detik |

Waktu di atas dibaca dari indikator eksekusi di tangkapan layar Colab pada satu kali percobaan, jadi hanya gambaran kasar.

### 2. Pembersihan data

- Memeriksa nilai `NULL` di semua kolom, lalu menghapus baris yang mengandung `NULL` dengan `dropna()`. Hasil pemeriksaan: tidak ada nilai kosong.
- Menyeragamkan huruf semua kolom dengan `initcap`. Data mentah punya penulisan yang tidak konsisten, misalnya `aAroN ADaMS`.
- Membulatkan `Billing Amount` menjadi 2 angka desimal.
- Mengubah `Age` menjadi integer.

### 3. Agregasi dan grouping

- Jumlah pasien per kondisi medis, per jenis kelamin, dan per hasil tes.
- Statistik deskriptif usia (mean, stddev, min, max).
- Rata-rata tagihan per penyedia asuransi.
- Total tagihan per rumah sakit, diurutkan dari terbesar.
- Tagihan maksimum per kondisi medis.

### 4. Klasifikasi: `Test Results`

Fitur: `Age`, `Billing Amount`, `Gender`, `Medical Condition`. Data dibagi 80% latih dan 20% uji (`seed=42`).

| Model | Detail |
|---|---|
| **Decision Tree Classifier** | `StringIndexer` + `VectorAssembler`, evaluasi accuracy dan confusion matrix |
| **Logistic Regression** | `Pipeline` (indexing, `StandardScaler`, assembler), `regParam=0.01` |

### 5. Regresi: `Billing Amount`

Fitur: `Age`, `Room Number`, `Gender`, `Medical Condition`. Split 80/20 (`seed=42`), dievaluasi dengan RMSE.

| Model | Detail |
|---|---|
| **Linear Regression** | Model dasar |
| **Random Forest Regressor** | `numTrees=100` |

### 6. Klastering: K-Means

Pasien dikelompokkan berdasarkan `Age` dengan `k=3`. Untuk setiap klaster ditampilkan statistik usia dan kondisi medis dominan.

### 7. Frequent Pattern Mining: FP-Growth

Transaksi dibentuk dari `Medical Condition`, `Admission Type`, dan `Medication`, dengan `minSupport=0.01` dan `minConfidence=0.05`. Keluarannya adalah *frequent itemsets* dan *association rules*.

## Ringkasan Hasil

**Distribusi data**

- Jumlah pasien: 55.500 (Female 27.726, Male 27.774).
- Usia: rata-rata ± 51,5 tahun (min 13, maks 89).
- Kondisi medis tersebar hampir merata di 6 kategori, sekitar 9.200 sampai 9.300 pasien per kategori.
- Hasil tes: Abnormal 18.627, Normal 18.517, Inconclusive 18.356.
- Rata-rata tagihan per asuransi berada di kisaran 25.389 sampai 25.616.

**Performa model**

| Tugas | Model | Metrik | Nilai |
|---|---|---|---|
| Klasifikasi | Decision Tree | Accuracy | 0,3267 |
| Klasifikasi | Logistic Regression | Accuracy | 0,3312 |
| Regresi | Linear Regression | RMSE | 14.198,58 |
| Regresi | Random Forest | RMSE | 14.198,16 |

**Klaster usia (K-Means, k=3)**

| Klaster | Jumlah pasien | Rentang usia | Rata-rata usia | Kondisi dominan |
|---|---|---|---|---|
| 0 | 18.624 | 13 – 40 | 28,97 | Arthritis, Obesity |
| 2 | 18.947 | 41 – 63 | 52,01 | Diabetes, Obesity |
| 1 | 17.929 | 64 – 89 | 74,49 | Arthritis, Hypertension |

**FP-Growth.** Kombinasi item tersebar cukup merata. Nilai *lift* pada aturan asosiasi yang ditemukan berada di sekitar 1 (kira-kira 0,92 sampai 1,08).

## Struktur Proyek

```
.
├── Presentasi_DATMIN.ipynb   # Notebook analisis (PySpark)
├── healthcare_rows.csv       # Dataset pasien (55.500 baris)
└── README.md
```

## Cara Menjalankan

### Prasyarat

- Python 3.8+
- Java 8/11/17 (dibutuhkan Spark)

### Instalasi

```bash
pip install pyspark pandas psycopg2-binary jupyter
```

### Menjalankan notebook

```bash
jupyter notebook Presentasi_DATMIN.ipynb
```

Atau upload notebook ke Google Colab.

### Sumber data

**Opsi A: dari CSV (paling mudah).** Ganti sel pemuatan data dengan:

```python
spark_df = spark.read.csv("healthcare_rows.csv", header=True, inferSchema=True)
```

**Opsi B: dari PostgreSQL.** Simpan connection string di environment variable dan jangan menulisnya langsung di kode:

```bash
export DATABASE_URL="postgresql://USER:PASSWORD@HOST:5432/postgres"
```

```python
import os, psycopg2, pandas as pd

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cursor = conn.cursor()
cursor.execute("SELECT * FROM healthcare;")
df = pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description])
cursor.close()
conn.close()
```

## Catatan dan Keterbatasan

- **Performa model rendah.** Accuracy klasifikasi sekitar 33% mendekati tebakan acak untuk 3 kelas yang seimbang. RMSE regresi (± 14.198) hampir sama dengan simpangan baku tagihan, jadi model regresi hampir hanya memprediksi nilai rata-rata. Fitur yang dipakai tampaknya belum punya hubungan yang kuat dengan target, dan data ini kemungkinan besar bersifat sintetis.
- **Klaster hanya berdasarkan usia**, sehingga kondisi medis dominan tiap klaster sangat mirip.
- **Perbandingan `local[1]` vs `local[2]`** hanya satu kali percobaan dan mencakup overhead startup Spark, jadi belum cukup untuk kesimpulan yang kuat.
- Kolom tanggal (`Date of Admission`, `Discharge Date`) belum dikonversi ke tipe `date`, dan kode konversinya masih dikomentari di notebook.
- Data dimuat lewat pandas lalu `createDataFrame`. Untuk data yang jauh lebih besar, sebaiknya baca langsung dengan `spark.read` (CSV/Parquet/JDBC).
