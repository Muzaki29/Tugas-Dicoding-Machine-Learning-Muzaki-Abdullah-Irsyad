# src/eda.py
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # gunakan backend non-GUI agar tidak butuh tkinter
import matplotlib.pyplot as plt
import os

# 1️⃣ Membaca dataset
df = pd.read_csv("data/Mall_Customers.csv")

# 2️⃣ Menampilkan informasi dataset
print("===== INFORMASI DATASET =====")
print(df.info())

# 3️⃣ Menampilkan 5 data teratas
print("\n===== 5 DATA TERATAS =====")
print(df.head())

# 4️⃣ Statistik deskriptif
print("\n===== STATISTIK DESKRIPTIF =====")
print(df.describe())

# 5️⃣ Mengecek missing value
print("\n===== JUMLAH NILAI KOSONG =====")
print(df.isnull().sum())

# 6️⃣ Visualisasi distribusi fitur penting
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
sns.histplot(df["Age"], kde=True, color="skyblue")
plt.title("Distribusi Umur")

plt.subplot(1, 3, 2)
sns.histplot(df["Annual Income (k$)"], kde=True, color="orange")
plt.title("Distribusi Pendapatan Tahunan")

plt.subplot(1, 3, 3)
sns.histplot(df["Spending Score (1-100)"], kde=True, color="green")
plt.title("Distribusi Spending Score")

plt.tight_layout()

# Pastikan folder output ada
os.makedirs("output", exist_ok=True)

# Simpan hasil visualisasi
output_path = "output/eda_output.png"
plt.savefig(output_path)
print(f"\n✅ Visualisasi berhasil disimpan di: {output_path}")
