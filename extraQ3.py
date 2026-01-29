#3)	To design and train the Hopfield net to map the input vector with the stored vector and correct them.
import numpy as np
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            p = p.reshape(-1, 1)
            self.weights += np.dot(p, p.T)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, steps=5):
        for _ in range(steps):
            for i in range(self.size):
                net_input = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if net_input >= 0 else 0
        return pattern
# Define training patterns
patterns = np.array([[1, 0, 1, 0, 1],
                        [0, 1, 0, 1, 0],
                        [1, 1, 0, 0, 1]])
# Create Hopfield network instance
hopfield_net = HopfieldNetwork(size=5)
# Train the Hopfield network
hopfield_net.train(patterns)
# Test the Hopfield network with a noisy pattern
test_pattern = np.array([1, 0, 0, 0, 1])  # Noisy version of [1, 0, 1, 0, 1]
recalled_pattern = hopfield_net.recall(test_pattern.copy())
print("Testing the trained Hopfield Network:")
print(f"Input Pattern: {test_pattern}, Recalled Pattern: {recalled_pattern}")
# Output:
# Testing the trained Hopfield Network:
# Input Pattern: [1 0 0 0 1], Recalled Pattern: [1 1 1 1 1]