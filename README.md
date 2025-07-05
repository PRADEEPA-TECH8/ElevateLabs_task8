# 🧠 K-Means Clustering on Mall Customer Dataset

## 📌 Task 8 - AI & ML Internship (Elevate Labs)

This project demonstrates the use of **K-Means Clustering**, an unsupervised machine learning technique, to segment mall customers into different groups based on their spending behavior and income level.

---

## 📂 Dataset

- **Source**: [Kaggle - Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **Features Used**:
  - Age
  - Annual Income (k$)
  - Spending Score (1-100)

---

## 🛠️ Tools & Libraries

- Python 3.10
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PCA (for 2D visualization)
- Silhouette Score (for evaluation)

---

## 📈 Steps Performed

1. **Loaded** the dataset (`Mall_Customers.csv`)
2. **Preprocessed** the data:
   - Encoded `Gender` as numerical
   - Dropped `CustomerID`
3. **Applied the Elbow Method** to find the optimal number of clusters
4. **Trained KMeans** with `k=5`
5. **Visualized** clusters using PCA
6. **Evaluated** the clustering with **Silhouette Score**

---

## 📊 Results

- ✅ **Optimal Clusters (K)**: 5
- ✅ **Silhouette Score**: `0.357`
- ✅ Cluster Centers revealed interesting customer segments like:
  - High-income high-spenders
  - Low-income high-spenders
  - Moderate-income low-spenders, etc.

---

## 📸 Visualization (PCA - 2D Cluster View)

_Include a screenshot of the plotted clusters here if possible._

---

## 📁 Files Included

- `kmeans_clustering_task8.py` – Full code implementation
- `Mall_Customers.csv` – Dataset used
- `clustered_customers.csv` – Output with cluster labels


