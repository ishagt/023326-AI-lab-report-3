#1)	To design and train a perceptron for identifying ODD and EVEN number.
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
# Define training data for ODD and EVEN number
training_inputs = np.array([[0, 0, 0, 0],  # 0 - EVEN
                                [0, 0, 0, 1],  # 1 - ODD
                                [0, 0, 1, 0],  # 2 - EVEN   
                                [0, 0, 1, 1],  # 3 - ODD
                                [0, 1, 0, 0],  # 4 - EVEN
                                [0, 1, 0, 1],  # 5 - ODD
                                [0, 1, 1, 0],  # 6 - EVEN
                                [0, 1, 1, 1],  # 7 - ODD
                                [1, 0, 0, 0],  # 8 - EVEN
                                [1, 0, 0, 1],  # 9 - ODD
                                [1, 0, 1, 0],  #10 - EVEN
                                [1, 0, 1, 1],  #11 - ODD
                                [1, 1, 0, 0],  #12 - EVEN
                                [1, 1, 0, 1],  #13 - ODD
                                [1, 1, 1, 0],  #14 - EVEN
                                [1, 1, 1, 1]]) #15 - ODD
labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
# Create perceptron instance    
perceptron = Perceptron(input_size=4)
# Train the perceptron
perceptron.train(training_inputs, labels, epochs=10)
# Test the perceptron
print("Testing the trained perceptron for ODD and EVEN number:")
for inputs in training_inputs:
    print(f"Input: {inputs}, Predicted Output: {perceptron.predict(inputs)}")
    
    #output:
# Testing the trained perceptron for ODD and EVEN number:
# Input: [0 0 0 0], Predicted Output: 0
# Input: [0 0 0 1], Predicted Output: 1
# Input: [0 0 1 0], Predicted Output: 0
# Input: [0 0 1 1], Predicted Output: 1
# Input: [0 1 0 0], Predicted Output: 0
# Input: [0 1 0 1], Predicted Output: 1
# Input: [0 1 1 0], Predicted Output: 0
# Input: [0 1 1 1], Predicted Output: 1
# Input: [1 0 0 0], Predicted Output: 0
# Input: [1 0 0 1], Predicted Output: 1
# Input: [1 0 1 0], Predicted Output: 0
# Input: [1 0 1 1], Predicted Output: 1
# Input: [1 1 0 0], Predicted Output: 0
# Input: [1 1 0 1], Predicted Output: 1
# Input: [1 1 1 0], Predicted Output: 0
# Input: [1 1 1 1], Predicted Output: 1
