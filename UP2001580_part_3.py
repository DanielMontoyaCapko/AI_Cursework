import numpy as np

# Sigmoid activation function.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Backpropagation process to update the weights of the network
def sigmoid_derivative(x):
    return x * (1 - x)

def preprocess_data():
    # Assume you have a dataset with features and corresponding labels
    # Replace this with your actual data loading and preprocessing logic
    # For demonstration purposes, let's use a small example dataset
    data = np.array([
        # Replace this with your actual feature values
        [0.1, 0.2, 0.3, 0.4, 0.5,],
        [0.6, 0.7, 0.8, 0.9, 1.0,],
        [0.18, 0.6, 0.33, 0.41, 0.6,],
        [0.77, 0.71, 0.3, 0.79, 0.94,],
        [0.65, 0.56, 0.74, 0.92, 0.22,],
        [0.92, 0.8, 0.16, 0.87, 0.54,],
    ])

    labels = np.array([
        [0],  # label for non-fraud
        [1],  # label for fraud
        [0],
        [1],
        [1],
        [0],
    ])

    return data, labels

def train_single_layer_network(inputs, targets, learning_rate, epochs):
    input_size = inputs.shape[1]
    output_size = targets.shape[1]

    # Initialize weights and biases
    weights = np.random.rand(input_size, output_size)
    biases = np.zeros((1, output_size))

    for epoch in range(epochs):
        # Forward pass
        layer_input = np.dot(inputs, weights) + biases
        layer_output = sigmoid(layer_input)

        # Calculate error
        error = targets - layer_output

        # Backpropagation
        gradient = error * sigmoid_derivative(layer_output)
        weights += learning_rate * np.dot(inputs.T, gradient)
        biases += learning_rate * np.sum(gradient, axis=0, keepdims=True)

    return weights, biases

def predict(inputs, weights, biases):
    return sigmoid(np.dot(inputs, weights) + biases)

# Load and preprocess data
data, labels = preprocess_data()

# Set hyperparameters
learning_rate = 0.01
epochs = 1000

# Train the network
trained_weights, trained_biases = train_single_layer_network(data, labels, learning_rate, epochs)

# Generate Test Data
def generate_random_data():
    return [np.random.rand() for _ in range(5)]

test_data = [generate_random_data() for _ in range(10)]

# Make predictions on the test data
predictions = predict(np.array(test_data), trained_weights, trained_biases)

# Print the results
print("Trained Weights:")
print(trained_weights)
print("\nTrained Biases:")
print(trained_biases)
print("\nPredictions for Test Data:")
for index, feature_values in enumerate(test_data):
    print(f"Test Data: {feature_values}, Prediction: {predictions[index][0]}")
