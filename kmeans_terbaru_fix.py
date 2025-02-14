# -*- coding: utf-8 -*-
"""kmeans_terbaru.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cmzshvLx_g4dSoQk7r5rs1scmPHs27Rm
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Mengunggah file CSV
uploaded = files.upload()

# Memuat dataset
data = pd.read_csv('tokopedia dataset 1.csv', sep=';')

# Menampilkan informasi data
print("Informasi dataset:")
print(data.info())

# Memilih fitur yang relevan untuk clustering
features = data[['Price', 'Number Sold', 'Total Review']]

# Membuat heatmap korelasi
plt.figure(figsize=(10, 6))
correlation_matrix = features.corr()  # Gunakan 'features' di sini
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap features')
plt.show()

# Menggunakan 'features' untuk data_features
data_features = features

# Menangani missing values jika ada
features = data.dropna()

# Menormalisasi fitur
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_features)

# Menentukan jumlah cluster optimal menggunakan metode elbow
wcss = []
range_n_clusters = range(1, 11)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plot hasil elbow method
plt.figure(figsize=(8, 5))
plt.plot(range_n_clusters, wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

# Membuat model KMeans dengan 4 cluster
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(data_features)

# Menghitung Silhouette Score
silhouette_avg = silhouette_score(data_features, clusters)

print(f'Silhouette Score untuk 4 cluster: {silhouette_avg}')

# Menambahkan label cluster ke dataset
data['Cluster'] = kmeans.labels_

# Mendefinisikan daftar fitur yang relevan
features_list = ['Price', 'Number Sold', 'Total Review']

# Membuat diagram untuk nilai rata-rata fitur berdasarkan cluster
mean_values = data.groupby('Cluster')[features_list].mean().reset_index()
mean_values = mean_values.melt(id_vars='Cluster', var_name='Fitur', value_name='Nilai Rata-rata')
print("Rata-rata features tiap cluster:")
print(mean_values)

# Mengurangi dimensi untuk visualisasi
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(data_scaled)

# Membuat DataFrame untuk hasil PCA
pca_df = pd.DataFrame(data=reduced_features, columns=['PC1', 'PC2'])
pca_df['Cluster'] = data['Cluster']

# Mengubah nilai cluster dari 0-3 menjadi 1-4
pca_df['Cluster'] = pca_df['Cluster'] + 1

# Pastikan kolom Cluster, PC1, dan PC2 ada
print(pca_df.columns)

# Periksa jumlah data untuk setiap cluster
print("\nJumlah data dalam setiap cluster:")
print(pca_df['Cluster'].value_counts())

# Periksa tipe data kolom Cluster
print("\nTipe data kolom Cluster:")
print(pca_df['Cluster'].dtypes)

# Plot hanya jika kolom dan cluster valid
if 'Cluster' in pca_df.columns and 'PC1' in pca_df.columns and 'PC2' in pca_df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='PC1', y='PC2', hue='Cluster', data=pca_df,
        palette='coolwarm', s=100, edgecolor='k'
    )
    plt.title('K-means Clustering with PCA Reduction')
    plt.xlabel('Main Component 1')
    plt.ylabel('Main Component 2')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()
else:
    print("Kolom yang diperlukan tidak ada dalam dataframe.")

# Cek nilai unik di kolom Cluster
print("Nilai unik dalam kolom Cluster:")
print(pca_df['Cluster'].unique())

# Cek tipe data kolom Cluster
print("\nTipe data kolom Cluster:")
print(pca_df['Cluster'].dtypes)

# Cek apakah Cluster 1 ada
if 1 in pca_df['Cluster'].values:
    print("\nCluster 1 terdeteksi.")
else:
    print("\nCluster 1 tidak terdeteksi.")

# Normalisasi nilai cluster agar dimulai dari 1
data['Cluster'] = data['Cluster'] - data['Cluster'].min() + 1

# Periksa nilai unik setelah normalisasi
print("Nilai unik setelah normalisasi:")
print(data['Cluster'].unique())

print("\nKategori untuk setiap cluster:")
for cluster in sorted(data['Cluster'].unique()):  # Gunakan nilai unik yang sudah dinormalisasi
    example_categories = data[data['Cluster'] == cluster]['Category'].unique()
    print(f"Cluster {cluster}: {example_categories}")

# Mengubah nilai cluster dari 2-5 menjadi 1-4
data['Cluster'] = data['Cluster'] - 1

# Set the style of seaborn
sns.set(style="whitegrid")

# Misalkan df sudah berisi kolom 'Price', 'Number Sold', 'Total Review', dan 'Cluster'

# Create a scatter plot for 'Number Sold' vs 'Price' with K-means clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Price', y='Number Sold', hue='Cluster', palette='coolwarm', s=100, edgecolor='k')
plt.title('K-means Clustering: Number Sold vs Price')
plt.xlabel('Price')
plt.ylabel('Number Sold')
plt.legend(title='Cluster')
plt.show()

# Create a scatter plot for 'Number Sold' vs 'Total Review' with K-means clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Total Review', y='Number Sold', hue='Cluster', palette='coolwarm', s=100, edgecolor='k')
plt.title('K-means Clustering: Number Sold vs Total Review')
plt.xlabel('Total Review')
plt.ylabel('Number Sold')
plt.legend(title='Cluster')
plt.show()

# Create a scatter plot for 'Price' vs 'Total Review' with K-means clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Total Review', y='Price', hue='Cluster', palette='coolwarm', s=100, edgecolor='k')
plt.title('K-means Clustering: Price vs Total Review')
plt.xlabel('Total Review')
plt.ylabel('Price')
plt.legend(title='Cluster')
plt.show()

print(data.columns)