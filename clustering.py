from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler()

X = np.array([[1, 2, 150], [1, 4, 10], [1, 0, 13],
              [10, 2, 48], [10, 4, 50], [10, 0, 160]])

scaler.fit(X)
X = scaler.transform(X)

kmeans = KMeans(n_clusters=3, random_state=0)

kmeans.fit(X)

print(kmeans.labels_)

X_pred = scaler.transform([[4, 3, 1], [20, 100, 50]])
kmeans.predict(X_pred)