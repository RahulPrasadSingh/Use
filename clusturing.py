# K-Means Clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('your_file.csv')
X = df.select_dtypes(include=['float64', 'int64'])  # Numeric columns only

# Create and fit model
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Add cluster labels to dataframe
df['Cluster'] = clusters

# Visualize (if 2D data)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='red', marker='X', label='Centroids')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.legend()
plt.show()

# Elbow method to find optimal K
inertias = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    inertias.append(model.inertia_)
plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


# DBSCAN Clustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('your_file.csv')
X = df.select_dtypes(include=['float64', 'int64'])

# Scale features (important for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit model
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

df['Cluster'] = clusters
print("Number of clusters:", len(set(clusters)) - (1 if -1 in clusters else 0))
print("Number of outliers:", list(clusters).count(-1))

# Visualize
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title('DBSCAN Clustering (-1 = outliers)')
plt.show()

# Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load data
df = pd.read_csv('your_file.csv')
X = df.select_dtypes(include=['float64', 'int64'])

# Create and fit model
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
clusters = agg.fit_predict(X)

df['Cluster'] = clusters

# Create dendrogram
linked = linkage(X, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(linked)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# GMM Clustering
from sklearn.mixture import GaussianMixture

# Load data
df = pd.read_csv('your_file.csv')
X = df.select_dtypes(include=['float64', 'int64'])

# Create and fit model
gmm = GaussianMixture(n_components=3, random_state=42)
clusters = gmm.fit_predict(X)
probabilities = gmm.predict_proba(X)  # Soft assignment

df['Cluster'] = clusters
print("BIC Score:", gmm.bic(X))  # Lower is better
print("AIC Score:", gmm.aic(X))  # Lower is better

# Visualize
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.show()
