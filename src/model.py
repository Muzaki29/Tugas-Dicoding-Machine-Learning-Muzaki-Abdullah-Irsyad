# src/model.py

import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')  # non-GUI backend
import matplotlib.pyplot as plt
import joblib

# === 1️⃣ Pastikan path ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # naik 1 level dari /src
DATA_PATH = os.path.join(BASE_DIR, "output", "preprocessed_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Buat folder output & models jika belum ada
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 2️⃣ Baca data hasil preprocessing ===
df = pd.read_csv(DATA_PATH)

# === 3️⃣ Tentukan jumlah cluster optimal (Elbow Method) ===
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df)
    inertia.append(kmeans.inertia_)

# Simpan grafik Elbow ke output/
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.title("Metode Elbow untuk Menentukan k Optimal")
elbow_path = os.path.join(OUTPUT_DIR, "elbow_method.png")
plt.savefig(elbow_path)
plt.close()

print(f"✅ Grafik Elbow Method disimpan di: {elbow_path}")

# === 4️⃣ Pilih jumlah cluster optimal (misalnya k=5) ===
k_optimal = 5
model = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(df)

# === 5️⃣ Simpan model KMeans ===
model_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")
joblib.dump(model, model_path)
print(f"✅ Model K-Means disimpan di: {model_path}")

# === 6️⃣ Simpan hasil clustering ke CSV ===
clustered_path = os.path.join(OUTPUT_DIR, "clustered_data.csv")
df.to_csv(clustered_path, index=False)
print(f"✅ File hasil clustering disimpan di: {clustered_path}")

# === 7️⃣ Visualisasi hasil clustering (2 fitur utama) ===
plt.figure(figsize=(8, 6))
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"],
            c=df["Cluster"], cmap="viridis", s=50)
plt.title("Visualisasi Klaster Pelanggan")
plt.xlabel("Annual Income (Standardized)")
plt.ylabel("Spending Score (Standardized)")
plt.colorbar(label="Cluster")
plt.tight_layout()

clusters_plot = os.path.join(OUTPUT_DIR, "customer_clusters.png")
plt.savefig(clusters_plot)
plt.close()

print(f"✅ Visualisasi klaster disimpan di: {clusters_plot}")

# === 8️⃣ Tampilkan ringkasan hasil ===
print("\n===== CONTOH HASIL CLUSTERING =====")
print(df.head())
