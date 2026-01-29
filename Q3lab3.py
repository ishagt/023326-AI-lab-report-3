#3)	To design and train a perceptron for NOT gate.
import numpy as np
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size + 1)  # +1 for bias weight
        self.learning_rate = learning_rate

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # bias weight
        return self.activation_function(summation)

    def train(self, training_inputs, labels, epochs):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error  # update bias weight
# Define training data for NOT gate
training_inputs = np.array([[0],
                                [1]])
labels = np.array([1, 0])
# Create perceptron instance
perceptron = Perceptron(input_size=1)
# Train the perceptron
perceptron.train(training_inputs, labels, epochs=10)
# Test the perceptron
print("Testing the trained perceptron for NOT gate:")
for inputs in training_inputs:
    print(f"Input: {inputs}, Predicted Output: {perceptron.predict(inputs)}")
    
    #output:
# Testing the trained perceptron for NOT gate:
# Input: [0], Predicted Output: 1
# Input: [1], Predicted Output: 0