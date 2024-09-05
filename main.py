import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Membaca Data Set "Kualitas Pisang" dari file CSV
data = pd.read_csv("./banana.csv")

# Menampilkan Data Awal
print(data)

# Menghapus Kolom/Feature 'Quality' dari Data Set (Kolom "Quality" Memiliki tipe data "string")
data = data.drop(columns=['Quality'])

# Mengisi data kosong dengan mean agar menghindari kekeliruan perhitungan
data = data.fillna(data.median())

# Mengubah DataSet ke Array Numpy
data_array = np.array(data)
print(data_array)

# Visualisasi data asli sebelum reduksi dimensi dengan PCA (menggunakan 2 kolom pertama ["Size" & "Weight"])
plt.scatter(data_array[:, 0], data_array[:, 1])
plt.title('Data Asli Sebelum Reduksi Dimensi')
plt.xlabel('Size')
plt.ylabel('Weight')
plt.show()

# 1. Mencari Mean (nilai rata-rata) dari data
mean = np.mean(data_array, axis=0)

# 2. Menghitung Zero Mean (setiap nilai pada data sampel dikurangi nilai rata-rata tiap parameter yang terkait)
zero_mean_data = data_array - mean


# 3. Membangun matriks Covarians dengan mengkalikan matriks Zero Mean dengan transposenya
cov_matrix = np.cov(zero_mean_data, rowvar=False)

# 4. Menghitung eigenvalue
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 5. Menghitung matriks eigenvektor
# Sudah dihitung pada langkah sebelumnya, tersimpan di 'eigenvectors'

# 6. Mengurangi dimensi N sebesar K dimensi yang didapatkan dari eigenvalue yang terbesar sampai yang terkecil
# Mengurutkan eigenvalue dan eigenvector
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
#
# Memilih K eigenvector teratas untuk mengurangi dimensi
K = 2  # Misalnya kita ingin mengurangi dimensi menjadi 2
reduced_eigenvectors = sorted_eigenvectors[:, :K]

# Mengurangi dimensi data
reduced_data = np.dot(zero_mean_data, reduced_eigenvectors)

# Visualisasi hasil reduksi jika direduksi ke 2D
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.title('Hasil Reduksi Dimensi dengan PCA')
plt.xlabel('Komponen Utama 1')
plt.ylabel('Komponen Utama 2')
plt.show()

# Menampilkan eigenvalues dan eigenvectors
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
print("Data setelah reduksi:\n", reduced_data)