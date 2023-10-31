# Laporan Proyek Machine Learning
### Nama : Moch Reki Hadiyanto 
### Nim : 211351083 
### Kelas : Pagi B

## Domain Proyek

Web App yang saya kembangkan ini sebaiknya digunakan oleh/berdampingan dengan seorang profesional agar variabel-variabel yang diinputkan tidak semena-mena dimasukkan begitu saja, Web App ini dikembangkan untuk memudahkan pengguna dalam menentukan proses pengobatan selanjutnya tergantung dari hasil output Web App ini. Namun jika anda bukanlah seorang profesional sebaiknya mendatangi langsung ahlinya.

## Business Understanding

Memungkinkan seorang profesional/dokter bekerja lebih cepat dan tepat, dengan itu lebih banyak pasien akan mendapatkan penanganan langsung dari seorang dokter.

### Problem Statements
- Semakin banyaknya orang yang didiagnosa mengidap diabetes dikarenakan pola hidup modern yang tidak teratur/buruk, maka semakin banyak pula pasien yang harus ditangani oleh ahli profesional

### Goals
- Memudahkan dokter/ahli profesional dalam menentukan pengobatan selanjutnya bagi pasien yang mengidap/tidak mengidap penyakit diabetes dengan hasil yang dikeluarkan oleh Web App.

## Data Understanding
Diabetes dataset adalah datasets yang saya gunakan, saya dapatkan dari kaggle.com. Data-data ini dapat digunakan untuk melakukan prediksi/diagnosa terhadap pasien yang bisa saja memiliki penyakit diabetes. Datasets ini mengandung 9 Attribut(Kolom) dan 768 data(baris) pada saat sebelum pemrosesan data cleasing dan EDA.
<br> 

[Diabetes dataset](https://www.kaggle.com/datasets/mahdiehhajian/diabetes).

### Variabel-variabel pada Diabetes dataset adalah sebagai berikut:
- Pregnancies : merupakan jumlah angka pasien hamil.
- Glucose    : merupakan level glucose pasien.
- BloodPressure : merupakan level tekanan darah pasien pada saat diperiksa.
- SkinThickness : merupakan ketebalan kulit pasien dalam satuan mm.
- Insulin : menunjukkan tingkat insulin pada tubuh pasien.
- BMI     : merupakan pengukuran lemak tubuh berdasarkan berat dan tinggi badan.
- DiabetesPedigreeFunction  : fungsi yang menghasilkan nilai pengaruh riwayat penyakit diabetes pada seseorang.
- Age : merupakan umur pasien.
- Outcome  : menunjukkan apakah pasien mengidap diabetes atau tidak.

## Data Preparation
Untuk data preparation ini saya melakukan EDA (Exploratory Data Analysis) terlebih dahulu, lalu melakukan proses data cleansing agar model yang dihasilkan memiliki score yang lebih tinggi.

Sebelum memulai data preparation, kita akan mendownload datasets dari kaggle yang akan kita gunakan, langkah pertama adalah memasukkan token kaggle,
``` bash
from google.colab import files
files.upload()
```
Lalu kita harus membuat folder untuk menampung file kaggle yang tadi telah diupload,
``` bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
Lalu, download datasets menggunakan code dibawah ini, 
``` bash
!kaggle datasets download -d mahdiehhajian/diabetes
```
Setelah download telah selesai, langkah selanjutnya adalah mengektrak file zipnya kedalam sebuah folder,
``` bash
!unzip diabetes.zip -d diabetes_prediction
!ls diabetes_prediction
```
Datasets telah diekstrak, seharusnya sekarang ada folder yang bernama diabetes_prediction dan di dalamnya terdapat file dengan ektensi .csv, <br>
Langkah selanjutnya adalah mengimport library yang dibutuhkan untuk melaksanakan data Exploration, data visualisation, dan data cleansing,
``` bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
Selanjutnya, mari baca file .csv yang tadi kita ekstrak, lalu melihat 3 data pertama yang ada pada datasets,
``` bash
data = pd.read_csv("diabetes_prediction/diabetes.csv")
data.head(3)
```
Lalu untuk melihat jumah data, mean data, data terkecil dan data terbesar bisa dengan kode ini,
``` bash
data.describe()
```
Untuk melihat typedata yang digunakan oleh masing-masing kolom bisa menggunakan kode ini,
``` bash
data.info()
```
Selanjutnya kita akan melihat korelasi antar kolomnya,
``` bash
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True)
```
![download](https://github.com/RexBaee/diabetes_prediction_2/assets/130348460/df4a6f8f-6791-44ba-aef8-a53cc7677771)
<br>
Korelasi antar kolom terlihat aman dan banyak kolom memiliki kaitan yang erat satu dengan yang lainnya.
selanjutnya kita akan lihat nilai dari masing masing kolom dan melihat apakah ada yang aneh/abnormal,
``` bash
plt.figure(figsize=(15,25))
for i in range(len(df.columns)):
    plt.subplot(521+i)
    sns.histplot(df.iloc[:,i])
plt.show()
```
![download](https://github.com/RexBaee/diabetes_prediction_2/assets/130348460/671e6038-3c5a-4665-8e25-a3a89a2dc757)
Seharusnya tidak mungkin BMI atau Blood Pressure untuk menjadi 0, kemungkinan terdapat sesuatu abnormal seperti peningkatan secara drastis, kita akan masukkan data median saya untuk mengisinya,
``` bash
median_BloodPressure = df['BloodPressure'].median()
df['BloodPressure'] = df['BloodPressure'].replace(0, median_BloodPressure)

median_BMI = df['BMI'].median()
df['BMI'] = df['BMI'].replace(0, median_BMI)
```
Saya juga tidak yakin apakah jika glucose, skin thickness dan insulin berangka 0, ianya akan menimbulkan masalah, kita akan coba selesaikan masalahnya,
``` bash
tmp_df = df.copy()
```
``` bash
TARGET = "Outcome"
plt.figure(figsize=(12,8))
plt.subplot(221)
sns.boxplot(x=df[TARGET],y=df["Glucose"])


plt.subplot(222)
sns.boxplot(x=df[TARGET],y=df["SkinThickness"])

plt.subplot(223)
sns.boxplot(x=df[TARGET],y=df["Insulin"])
plt.show()
```
![download](https://github.com/RexBaee/diabetes_prediction_2/assets/130348460/6d8c18b1-4c52-4ad7-be89-cd912ae0d3b6) <br>
``` bash
median_Glucose = tmp_df['Glucose'].median()
median_SkinThickness = tmp_df['SkinThickness'].median()
median_Insulin = tmp_df['Insulin'].median()
tmp_df['Glucose'] = tmp_df['Glucose'].replace(0, median_Glucose)
tmp_df['SkinThickness'] = tmp_df['SkinThickness'].replace(0, median_SkinThickness)
tmp_df['Insulin'] = tmp_df['Insulin'].replace(0, median_Insulin)

plt.figure(figsize=(12,8))
plt.subplot(221)
sns.boxplot(x=tmp_df[TARGET],y=tmp_df["Glucose"])

plt.subplot(222)
sns.boxplot(x=tmp_df[TARGET],y=tmp_df["SkinThickness"])

plt.subplot(223)
sns.boxplot(x=tmp_df[TARGET],y=tmp_df["Insulin"])
plt.show()
```
![download](https://github.com/RexBaee/diabetes_prediction_2/assets/130348460/b309f145-9aeb-4a49-91ea-f5f173aca883)
Dari plot diatas, kurang terlihat ya perbedaannya tapi ada perbedaannya, mari gunakan korelasi data saja untuk melihatnya lebih jelas,
``` bash
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()
```
![download](https://github.com/RexBaee/diabetes_prediction_2/assets/130348460/90a82493-7d33-409d-8f8c-c5579e5a9530) <br>
``` bash
plt.figure(figsize=(8, 6))
sns.heatmap(tmp_df.corr(), annot=True, fmt=".2f")
plt.show()
```
![download](https://github.com/RexBaee/diabetes_prediction_2/assets/130348460/886aa722-2a30-48dd-8ace-9b135abe3ede)
Disini kita bisa melihat data hasil prosesan kita (tmp_df) lebih tinggi korelasinya dibandingkan data biasanya (df), maka kita akan gunakan tmp_df,
```
x = tmp_df.drop("Outcome",axis=1)
y = tmp_df.Outcome
```
Pemisahan fitur dan target sudah dilakukan, mari lanjut dengan tahap modeling.

## Modeling
Model machine learning yang akan digunakan disini adalah logistic regression, langkah pertama yang harus kita lakukan adalah memasukkan semua library yang akan digunakan pada saat proses pembuatan model,
``` bash
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score,confusion_matrix
```
Langkah selanjutnya adalah membuat train test split, dengan persentase 30% test dan 70% train
``` bash
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
```
Lalu kita akan reshape bentuk arraynya,
``` bash
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)
```
Dan selanjutnya adalah mengimplementasikan model Logistic Regression dan melihat tingkat akurasinya,
``` bash
logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 200,)
print("test accuracy: {} ".format(logreg.fit(x_train, y_train).score(x_test, y_test)))
print("train accuracy: {} ".format(logreg.fit(x_train, y_train).score(x_train, y_train)))
```
Score yang kita dapatkan adalah 74.45% untuk test dan 78.77% untuk train, lalu akhirnya kita akan uji dengan data inputan kita sendiri,
``` bash
data = np.array([[3, 100, 80, 26, 115, 31.6, 0.503, 20]])
print(logreg.predict(data))
```
Dan hasilnya adalah 0 yang artinya tidak berpotensi mengidap diabetes. Sebelum mengakhiri ini, kita harus ekspor modelnya menggunakan pickle agar nanti bisa digunakan pada media lain.
``` bash
import pickle
filename = "prediksi_diabetes.sav"
pickle.dump(logreg,open(filename,'wb'))
```

## Evaluation
Matrik evaluasi yang saya gunakan disini adalah confusion matrix, karena ianya sangat cocok untuk kasus pengkategorian seperti kasus ini. Dengan membandingkan nilai aktual dengan nilai prediksi, kita bisa melihat jumlah hasil prediksi saat model memprediksi diabetes dan nilai aktual pun diabetes, serta melihat saat model memprediksi diabetes sedangkan data aktualnya tidak diabetes.
``` bash
y_pred = logreg.fit(x_train, y_train).predict(x_test)
cm = confusion_matrix(y_test,y_pred)

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['diabetes','not_diabetes']); ax.yaxis.set_ticklabels(['diabetes','not_diabetes']);
```
![download](https://github.com/RexBaee/diabetes_prediction_2/assets/130348460/9dad49b4-8dbe-425b-9ede-3fa88832cfb3)
<br>
Disitu terlihat jelas bahwa model kita berhasil memprediksi nilai diabetes yang sama dengan nilai aktualnya sebanyak 123 data.

## Deployment
[Diabetes Prediction App Reki](https://diabetesprediction2-reki.streamlit.app/)
![image](https://github.com/RexBaee/diabetes_prediction_2/assets/130348460/b95a65e6-4f9e-4298-be18-a9a9b4781304)



