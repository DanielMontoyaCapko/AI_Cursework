import numpy as np

def perceptron(input_data, weights, bias):
    """Perceptron function for binary classification."""
    weighted_sum = np.dot(input_data, weights) + bias
    output = 1 if weighted_sum > 0 else 0
    return output

def train_perceptron(input_data, labels, learning_rate=0.1, epochs=1000):
    """Train a perceptron using the perceptron learning algorithm."""
    input_size = input_data.shape[1]
    weights = np.random.rand(input_size)
    bias = np.random.rand()

    for epoch in range(epochs):
        for i in range(len(input_data)):
            prediction = perceptron(input_data[i], weights, bias)
            error = labels[i] - prediction

            # Update weights and bias
            weights += learning_rate * error * input_data[i]
            bias += learning_rate * error

    return weights, bias

def demonstrate_or_problem():
    """Demonstrate the solution of the OR problem."""
    # OR problem training data
    input_data_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels_or = np.array([0, 1, 1, 1])

    # Train the perceptron
    weights_or, bias_or = train_perceptron(input_data_or, labels_or)

    # Test the trained perceptron on OR problem
    for i in range(len(input_data_or)):
        prediction = perceptron(input_data_or[i], weights_or, bias_or)
        print(f"OR Problem - Input: {input_data_or[i]}, Prediction: {prediction}")

def demonstrate_xor_problem():
    """Demonstrate the inability to solve the XOR problem."""
    # XOR problem training data
    input_data_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels_xor = np.array([0, 1, 1, 0])

    # Train the perceptron
    weights_xor, bias_xor = train_perceptron(input_data_xor, labels_xor)

    # Test the trained perceptron on XOR problem
    for i in range(len(input_data_xor)):
        prediction = perceptron(input_data_xor[i], weights_xor, bias_xor)
        print(f"XOR Problem - Input: {input_data_xor[i]}, Prediction: {prediction}")

# Demonstrate the solution of the OR problem
print("Solution of the OR problem:")
demonstrate_or_problem()

# Demonstrate the inability to solve the XOR problem
print("\nInability to solve the XOR problem:")
demonstrate_xor_problem()
