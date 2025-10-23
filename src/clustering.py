# src/clustering.py

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # gunakan backend non-GUI agar aman di Windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# -----------------------------
# 1️⃣ Tentukan path dataset hasil preprocessing
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'output', 'preprocessed_data.csv')

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File tidak ditemukan: {DATA_PATH}")

# -----------------------------
# 2️⃣ Baca dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
print("===== DATASET SETELAH PREPROCESSING =====")
print(df.head())

# -----------------------------
# 3️⃣ Terapkan KMeans clustering
# -----------------------------
# Misal kita ingin 5 klaster
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df)

print("\n===== Hasil Cluster =====")
print(df['Cluster'].value_counts())

# -----------------------------
# 4️⃣ Visualisasi hasil klaster (2 fitur utama: Age vs Spending Score)
# -----------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x='Age',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set2',
    s=100
)
plt.title("Hasil KMeans Clustering")
plt.xlabel("Age (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.legend(title='Cluster')
plt.tight_layout()

# -----------------------------
# 5️⃣ Simpan visualisasi
# -----------------------------
output_folder = os.path.join(PROJECT_ROOT, 'output')
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'kmeans_clusters.png')
plt.savefig(output_path)
print(f"\n✅ Visualisasi klaster berhasil disimpan di: {output_path}")
