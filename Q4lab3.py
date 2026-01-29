
# Implement KNN algorithm
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """
        In KNN, 'fitting' is just storing the training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict labels for a list of data points.
        """
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        """
        Helper function to predict a single point.
        """
        # 1. Calculate Euclidean distances from 'x' to all training points
        # (x - self.X_train) calculates the difference for every point at once
        distances = np.sqrt(np.sum((x - self.X_train)**2, axis=1))

        # 2. Get indices of the k smallest distances
        k_indices = np.argsort(distances)[:self.k]

        # 3. Get the labels associated with those indices
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 4. Return the most common label (majority vote)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# --- Testing it out ---
if __name__ == "__main__":
    # Sample Data: [Feature 1, Feature 2]
    X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    y_train = np.array(['A', 'A', 'B', 'B', 'A', 'B'])

    model = KNN(k=3)
    model.fit(X_train, y_train)

    # Test point: [2, 2] - should be close to group 'A'
    test_points = np.array([[2, 2], [7, 7]])
    predictions = model.predict(test_points)

    print(f"Predictions: {predictions}")
    
    