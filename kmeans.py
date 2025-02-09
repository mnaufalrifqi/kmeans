import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Judul Aplikasi
st.title("K-Means Clustering Visualization")

# Mengunggah file CSV
uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

if uploaded_file is not None:
    # Memuat dataset
    data = pd.read_csv(uploaded_file, sep=';')
    
    # Menampilkan informasi dataset
    st.subheader("Informasi Dataset")
    st.write(data.info())
    
    # Memilih fitur yang relevan untuk clustering
    features = data[['Price', 'Number Sold', 'Total Review']]
    
    # Membuat heatmap korelasi
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    correlation_matrix = features.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={"shrink": .8})
    st.pyplot(plt)
    
    # Menangani missing values
    features = features.dropna()
    
    # Menormalisasi fitur
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(features)
    
    # Menentukan jumlah cluster optimal menggunakan metode elbow
    wcss = []
    range_n_clusters = range(1, 11)
    
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)
    
    # Plot hasil elbow method
    st.subheader("Elbow Method")
    plt.figure(figsize=(8, 5))
    plt.plot(range_n_clusters, wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    st.pyplot(plt)
    
    # Membuat model KMeans dengan 4 cluster
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)
    
    # Menghitung Silhouette Score
    silhouette_avg = silhouette_score(data_scaled, clusters)
    st.write(f"Silhouette Score untuk 4 cluster: {silhouette_avg}")
    
    # Menambahkan label cluster ke dataset
    data['Cluster'] = kmeans.labels_
    # Mendefinisikan daftar fitur yang relevan
    features_list = ['Price', 'Number Sold', 'Total Review']
    
    # Membuat diagram untuk nilai rata-rata fitur berdasarkan cluster
    mean_values = data.groupby('Cluster')[features_list].mean().reset_index()
    mean_values = mean_values.melt(id_vars='Cluster', var_name='Fitur', value_name='Nilai Rata-rata')
    st.subheader("Rata-rata Features Tiap Cluster")
    st.write(mean_values)

    # Mengurangi dimensi untuk visualisasi
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(data_scaled)

    # Membuat DataFrame untuk hasil PCA
    pca_df = pd.DataFrame(data=reduced_features, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = data['Cluster'] + 1  # Normalisasi cluster agar dimulai dari 1

    # Visualisasi PCA dengan cluster
    st.subheader("K-means Clustering with PCA Reduction")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='coolwarm', s=100, edgecolor='k')
    plt.xlabel('Main Component 1')
    plt.ylabel('Main Component 2')
    plt.legend(title='Cluster')
    plt.grid(True)
    st.pyplot(plt)
    
    # Scatter plot untuk fitur yang dipilih
    st.subheader("Scatter Plots for Clustering")
    
    # Scatter plot Number Sold vs Price
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Price', y='Number Sold', hue='Cluster', palette='coolwarm', s=100, edgecolor='k')
    plt.title('K-means Clustering: Number Sold vs Price')
    plt.xlabel('Price')
    plt.ylabel('Number Sold')
    plt.legend(title='Cluster')
    st.pyplot(plt)
    
    # Scatter plot Number Sold vs Total Review
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Total Review', y='Number Sold', hue='Cluster', palette='coolwarm', s=100, edgecolor='k')
    plt.title('K-means Clustering: Number Sold vs Total Review')
    plt.xlabel('Total Review')
    plt.ylabel('Number Sold')
    plt.legend(title='Cluster')
    st.pyplot(plt)
    
    # Scatter plot Price vs Total Review
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Total Review', y='Price', hue='Cluster', palette='coolwarm', s=100, edgecolor='k')
    plt.title('K-means Clustering: Price vs Total Review')
    plt.xlabel('Total Review')
    plt.ylabel('Price')
    plt.legend(title='Cluster')
    st.pyplot(plt)
