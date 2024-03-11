inertia = []
k_max = 10  # Số cụm
for k in range(1, k_max+1):
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model.fit(X)
    inertia.append(kmeans_model.inertia_)

# Vẽ biểu đồ Elbow
plt.plot(range(1, k_max+1), inertia, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, k_max+1):
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model.fit(X)
    labels = kmeans_model.labels_
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)

# Vẽ biểu đồ Silhouette
plt.plot(range(2, k_max+1), silhouette_scores, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()