# src/classification.py

import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(PROJECT_ROOT, 'output', 'clustered_data.csv')

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"File CSV tidak ditemukan: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print("===== DATASET SETELAH CLUSTERING =====")
print(df.head())
print("\n===== Distribusi Cluster =====")
print(df['Cluster'].value_counts())
