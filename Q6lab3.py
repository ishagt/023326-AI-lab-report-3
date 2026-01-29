# Implement K-Mean algorithm
import numpy as np
class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        # Randomly initialize centroids
        random_indices = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # Assign clusters
            self.labels = self._assign_clusters(X)

            # Store old centroids for convergence check
            old_centroids = self.centroids.copy()

            # Update centroids
            self.centroids = self._update_centroids(X)

            # Check for convergence
            if np.all(np.abs(self.centroids - old_centroids) < self.tol):
                break

    def _assign_clusters(self, X):
        distances = np.array([[np.linalg.norm(x - centroid) for centroid in self.centroids] for x in X])
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X):
        new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])
        return new_centroids

    def predict(self, X):
        return self._assign_clusters(X)
# --- Testing it out ---
if __name__ == "__main__":
    # Sample Data: [Feature 1, Feature 2]
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

    kmeans = KMeans(k=2)
    kmeans.fit(X)

    print("Centroids:")
    print(kmeans.centroids)

    print("Labels:")
    print(kmeans.labels)
    
    # Test prediction
    test_points = np.array([[2, 2], [7, 7]])
    predictions = kmeans.predict(test_points)
    print(f"Predictions for test points {test_points}: {predictions}")

# Output:
#     Centroids:
# [[7.33333333 9.        ]
#  [1.16666667 1.46666667]]
# Labels:
# [1 1 0 0 1 0]
# Predictions for test points [[2 2]
#  [7 7]]: [1 0]
