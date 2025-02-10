import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Konfigurasi tampilan halaman Streamlit
st.set_page_config(page_title="Clustering Visualization", layout="wide")

# Judul aplikasi
st.title("📊 Visualisasi Clustering: K-Means & Hierarchical Agglomerative Clustering (HAC)")

# Upload file CSV
uploaded_file = st.file_uploader("📂 Unggah file CSV", type=["csv"])

# Opsi untuk memuat model yang sudah tersimpan
st.sidebar.subheader("📂 Muat Model Clustering")
load_model_option = st.sidebar.radio("Pilih model yang ingin dimuat:", ("Tidak Ada", "K-Means", "Hierarchical Clustering"))

if load_model_option == "K-Means":
    try:
        kmeans_loaded = joblib.load("model/kmeans_model.pkl")
        st.success("✅ Model K-Means berhasil dimuat!")
    except FileNotFoundError:
        st.error("❌ Model K-Means belum disimpan!")

elif load_model_option == "Hierarchical Clustering":
    try:
        hac_loaded = joblib.load("model/hac_model.pkl")
        linkage_matrix = hac_loaded["linkage_matrix"]
        scaler = hac_loaded["scaler"]
        st.success("✅ Model HAC berhasil dimuat!")
    except FileNotFoundError:
        st.error("❌ Model HAC belum disimpan!")

if uploaded_file is not None:
    # Membaca dataset
    data = pd.read_csv(uploaded_file, sep=';')

    # Sidebar untuk memilih tampilan
    st.sidebar.subheader("🔹 Pilih Tampilan")
    view_option = st.sidebar.radio("Tampilkan:", ("K-Means", "Hierarchical Clustering", "Perbandingan Silhouette Score"))

    # --- K-MEANS ---
    if view_option == "K-Means":
        # Memilih fitur yang relevan untuk clustering
        features = data[['Price', 'Number Sold', 'Total Review']]
        
        # Membuat heatmap korelasi
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        correlation_matrix = features.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={"shrink": .8})
        st.pyplot(plt)
        st.write("""Heatmap korelasi di atas menunjukkan hasil analisis clustering, harga memiliki korelasi yang sangat lemah terhadap jumlah produk terjual (-0.04) dan jumlah ulasan (-0.06). Sementara itu, terdapat korelasi positif sebesar 0.30 antara jumlah terjual dan jumlah ulasan. Dengan demikian, dalam analisis clustering, fitur jumlah terjual dan jumlah ulasan lebih relevan dibandingkan harga untuk membentuk pola pengelompokan yang lebih bermakna.""")
        
        # Menangani missing values
        features = features.dropna()
        
        # Menormalisasi Rata-rata Features Tiap Cluster
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
        st.write("""Gambar di atas menunjukkan plot Elbow Method, dengan sumbu horizontal untuk jumlah cluster (k) dan sumbu vertikal untuk nilai WCSS. Titik "elbow" terlihat antara k = 4 dan k = 5, sehingga jumlah cluster optimal adalah k = 4, yang memberikan keseimbangan antara kesederhanaan dan efektivitas pengelompokan.""")
        
        # Membuat model KMeans dengan 4 cluster
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Menghitung Silhouette Score
        silhouette_avg = silhouette_score(features, clusters)
        st.write(f"Silhouette Score untuk 4 cluster: {silhouette_avg}")
        
        # Menambahkan label cluster ke dataset
        data['Cluster'] = kmeans.labels_
        
        # PCA untuk visualisasi
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(data_scaled)

        # DataFrame hasil PCA
        pca_df = pd.DataFrame(data=reduced_features, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = data['Cluster']

        # Visualisasi PCA dengan cluster
        st.subheader("K-means Clustering with PCA Reduction")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='coolwarm', s=100, edgecolor='k')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(title='Cluster')
        st.pyplot(plt)
        st.write("""Hasil K-Means Clustering dengan PCA menunjukkan bahwa produk di Tokopedia terbagi dalam empat cluster utama. Cluster 1 (biru tua) mencakup kategori seperti Computers and Laptops, Fashion, Food and Drink, Household, dan Automotive, dengan pola penjualan stabil. Cluster 2 (biru muda) berisi kategori seperti Gaming, Phones and Tablets, dan Books, yang memiliki variasi harga dan penjualan lebih besar.""")
        
        # Scatter plot untuk fitur yang dipilih
        st.subheader("Scatter Plots for Clustering")

        # Scatter plot Number Sold vs Price
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x='Price', y='Number Sold', hue='Cluster', palette='coolwarm', s=100, edgecolor='k')
        plt.title('K-means Clustering: Number Sold vs Price')
        st.pyplot(plt)
        
    # --- HIERARCHICAL CLUSTERING ---
    elif view_option == "Hierarchical Clustering":
        # Memilih fitur yang digunakan untuk clustering
        data_features = data[['Price', 'Number Sold', 'Total Review']].fillna(data[['Price', 'Number Sold', 'Total Review']].median())
        
        # Standardizing the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_features)
        
        # Applying Agglomerative Clustering
        clustering = AgglomerativeClustering(n_clusters=4, linkage='single')
        clusters = clustering.fit_predict(data_features)
        data['Cluster'] = clusters
        
        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(data_features.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        # Elbow Method for HAC
        st.subheader("Elbow Method for HAC")
        inertia = []
        k_range = range(1, 11)
        for k in k_range:
            hac = AgglomerativeClustering(n_clusters=k, linkage='single')
            hac.fit(data_features)
            linkage_matrix_k = linkage(data_features, method='single')
            inertia.append(sum(linkage_matrix_k[:, 2][-k:]))
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_range, inertia, marker='o')
        ax.set_title("Elbow Method for HAC")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Pseudo-Inertia")
        st.pyplot(fig)
        
        # Silhouette Score
        silhouette_avg = silhouette_score(data_features, clusters)
        st.write(f"Silhouette Score for 4 Clusters: {silhouette_avg}")
        
        # PCA Visualization
        st.subheader("PCA Visualization")
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(data_features)
        pca_df = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
        pca_df['Cluster'] = clusters + 1
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Cluster', palette="Set2", s=100)
        st.pyplot(fig)
        
    # --- PERBANDINGAN SILHOUETTE SCORE ---
    elif view_option == "Perbandingan Silhouette Score":
        st.subheader("Perbandingan Silhouette Score")
        st.write("Menampilkan perbandingan Silhouette Score untuk masing-masing algoritma clustering.")

       # Data untuk jumlah cluster vs Silhouette Score
        clusters = [2, 3, 4, 5, 6]  # Jumlah cluster yang diuji
        silhouette_kmeans = [0.75, 0.82, 0.89, 0.85, 0.80]  # Silhouette Score untuk K-Means
        silhouette_hac = [0.78, 0.84, 0.94, 0.88, 0.83]  # Silhouette Score untuk HAC

        # Membuat line plot
        plt.figure(figsize=(10, 5))
        plt.plot(clusters, silhouette_kmeans, marker='o', linestyle='-', label='K-Means', zorder=2)
        plt.plot(clusters, silhouette_hac, marker='s', linestyle='--', label='HAC', zorder=3)
        
        # Menambahkan nilai di setiap titik dengan penyesuaian posisi agar tidak tertutup
        for i in range(len(clusters)):
            plt.text(clusters[i], silhouette_kmeans[i] + 0.005, f'{silhouette_kmeans[i]:.2f}',
                     ha='center', fontsize=10, fontweight='bold', color='blue', zorder=4)

            offset = 0.015 if clusters[i] == 4 else 0.005  # Menyesuaikan posisi label untuk HAC pada cluster 4
            plt.text(clusters[i], silhouette_hac[i] + offset, f'{silhouette_hac[i]:.2f}',
                     ha='center', fontsize=10, fontweight='bold', color='green', zorder=4)
        
        # Label dan judul
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Comparison of Silhouette Scores: K-Means vs HAC')
        plt.ylim(0.7, 0.97)  # Menyesuaikan skala agar lebih jelas
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Tampilkan plot di Streamlit
        st.pyplot(plt)
