import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Title of the app
st.title("K-Means Clustering Visualization")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, sep=';')
    
    st.write("Dataset Information:")
    st.write(data.info())

# Selecting relevant features
features = data[['Price', 'Number Sold', 'Total Review']]

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(features.corr(), annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={"shrink": .8}, ax=ax)
st.pyplot(fig)

# Handling missing values
features = features.dropna()

# Standardizing the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(features)

# Elbow Method
st.subheader("Elbow Method")
wcss = []
range_n_clusters = range(1, 11)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range_n_clusters, wcss, marker='o')
ax.set_title("Elbow Method")
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("WCSS (Within-Cluster Sum of Squares)")
st.pyplot(fig)

# Applying KMeans with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Silhouette Score
silhouette_avg = silhouette_score(data_scaled, clusters)
st.write(f"Silhouette Score for 4 Clusters: {silhouette_avg}")

# Assign clusters to data
data['Cluster'] = kmeans.labels_

# PCA Visualization
st.subheader("PCA Visualization")
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(data_scaled)
pca_df = pd.DataFrame(reduced_features, columns=['PC1', 'PC2'])
pca_df['Cluster'] = data['Cluster']

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='coolwarm', s=100, edgecolor='k', ax=ax)
ax.set_title("K-means Clustering with PCA Reduction")
st.pyplot(fig)

# Scatter plots
st.subheader("Number Sold vs Price")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=data, x='Price', y='Number Sold', hue='Cluster', palette='coolwarm', s=100, edgecolor='k', ax=ax)
ax.set_title("K-means Clustering: Number Sold vs Price")
st.pyplot(fig)

st.subheader("Number Sold vs Total Review")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=data, x='Total Review', y='Number Sold', hue='Cluster', palette='coolwarm', s=100, edgecolor='k', ax=ax)
ax.set_title("K-means Clustering: Number Sold vs Total Review")
st.pyplot(fig)

st.subheader("Price vs Total Review")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=data, x='Total Review', y='Price', hue='Cluster', palette='coolwarm', s=100, edgecolor='k', ax=ax)
ax.set_title("K-means Clustering: Price vs Total Review")
st.pyplot(fig)
