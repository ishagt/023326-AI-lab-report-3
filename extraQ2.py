#2)	To create a Bi-directional Associative Memory (BAM) for ID and telephone number.
import numpy as np
class BAM:
    def __init__(self, input_size, output_size):
        self.weights = np.zeros((input_size, output_size))

    def train(self, input_patterns, output_patterns):
        for x, y in zip(input_patterns, output_patterns):
            self.weights += np.outer(x, y)

    def recall(self, input_pattern, steps=5):
        y = np.dot(input_pattern, self.weights)
        y = self.activation_function(y)
        for _ in range(steps):
            x = np.dot(y, self.weights.T)
            x = self.activation_function(x)
            y = np.dot(x, self.weights)
            y = self.activation_function(y)
        return x, y

    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)
# Define training data for BAM
input_patterns = np.array([[1, 0, 0, 1],  # ID 1
                           [0, 1, 1, 0],  # ID 2
                           [1, 1, 0, 0]]) # ID 3
output_patterns = np.array([[1, 0, 1, 0, 0, 1],  # Phone 1
                            [0, 1, 0, 1, 1, 0],  # Phone 2
                            [1, 1, 0, 0, 1, 0]]) # Phone 3
# Create BAM instance
bam = BAM(input_size=4, output_size=6)
# Train the BAM
bam.train(input_patterns, output_patterns)
# Test the BAM
test_input = np.array([1, 0, 0, 1])  # Test with ID 1
recalled_id, recalled_phone = bam.recall(test_input)
print("Testing the trained BAM for ID and Telephone number:")
print(f"Input ID Pattern: {test_input}, Recalled Phone Pattern: {recalled_phone}")
# Output:
# Testing the trained BAM for ID and Telephone number:
# Input ID Pattern: [1 0 0 1], Recalled Phone Pattern: [1 1 1 1 1 1]
