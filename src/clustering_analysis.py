import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load hasil clustering
df = pd.read_csv('../data/clustered_data.csv')

# Analisis deskriptif tiap cluster
print(df.groupby('Cluster').mean())

# Visualisasi hasil cluster
sns.pairplot(df, hue='Cluster', palette='tab10')
plt.show()

# Analisis distribusi pendapatan dan pengeluaran
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='tab10')
plt.title('Hasil Clustering Pelanggan')
plt.show()
