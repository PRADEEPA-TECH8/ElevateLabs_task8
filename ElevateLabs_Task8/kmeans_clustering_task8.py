# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 1: Load dataset
df = pd.read_csv("Mall_Customers.csv")
print("✅ Dataset Loaded Successfully!")
print(df.head())

# Step 2: Preprocessing
# Encode 'Gender' if present
if 'Gender' in df.columns:
    
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

# Drop CustomerID for clustering
X = df.drop('CustomerID', axis=1)

# Step 3: Elbow Method to find optimal K
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8,5))
plt.plot(K, inertia, marker='o')
plt.title('Elbow Method to determine Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Step 4: Apply KMeans with chosen K
optimal_k = 5  # Based on Elbow Curve
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(X)
df['Cluster'] = labels

# Step 5: PCA for 2D Visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X)

# Plot clusters
plt.figure(figsize=(8,6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='Set2', s=60)
plt.title("Customer Segments (K-Means Clustering)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Step 6: Evaluate with Silhouette Score
score = silhouette_score(X, labels)
print(f"🧠 Silhouette Score: {score:.3f}")

# Optional: View clustered data
print("\n📊 Sample Clustered Data:")
print(df.groupby('Cluster').mean())

# Optional: Save to CSV
df.to_csv("clustered_customers.csv", index=False)
