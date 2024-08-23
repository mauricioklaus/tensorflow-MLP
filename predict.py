import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf

def load_model(path):
    # Load weights and biases from the .npz file
    data = np.load(path)
    weights1 = tf.Variable(data['weights1'])
    biases1 = tf.Variable(data['biases1'])
    weights2 = tf.Variable(data['weights2'])
    biases2 = tf.Variable(data['biases2'])
    final_weights = tf.Variable(data['final_weights'])
    final_biases = tf.Variable(data['final_biases'])
    return weights1, biases1, weights2, biases2, final_weights, final_biases

def predict(weights1, biases1, weights2, biases2, final_weights, final_biases, inputs):
    # Define the activation function
    def neuron(x, weights, biases):
        return tf.add(tf.matmul(x, weights), biases)

    # Compute the model output
    layer1 = tf.nn.relu(neuron(inputs, weights1, biases1))
    layer2 = tf.nn.relu(neuron(layer1, weights2, biases2))
    output = tf.sigmoid(neuron(layer2, final_weights, final_biases))
    return output

# Load the model
weights1, biases1, weights2, biases2, final_weights, final_biases = load_model('saved_model.npz')

# Test input
test_input = tf.constant([[0.505451, -1.3482525, 0., 0., 0., 1.]], dtype=tf.float32)

# Make predictions
predictions = predict(weights1, biases1, weights2, biases2, final_weights, final_biases, test_input)
print(predictions.numpy())

