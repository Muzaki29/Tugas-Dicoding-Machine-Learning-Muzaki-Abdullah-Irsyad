# src/preprocessing.py

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Path dataset
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'Mall_Customers.csv')

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"File CSV tidak ditemukan: {DATA_FILE}")

# Baca dataset
df = pd.read_csv(DATA_FILE)

# Hapus kolom yang tidak relevan
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

# Encoding kolom Gender
if "Gender" in df.columns:
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])  # Female=0, Male=1

# Scaling fitur numerik
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

df_scaled = pd.DataFrame(scaled_features, columns=df.columns)

# Simpan hasil preprocessing
output_folder = os.path.join(PROJECT_ROOT, 'output')
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'preprocessed_data.csv')

if os.path.exists(output_path):
    os.remove(output_path)

df_scaled.to_csv(output_path, index=False)
print("\n✅ File hasil preprocessing disimpan di:", output_path)
