import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def save_model(weights1, biases1, weights2, biases2, final_weights, final_biases, path):
    # Create a dictionary to store weights and biases
    model_dict = {
        'weights1': weights1.numpy(),
        'biases1':  biases1.numpy(),
        'weights2': weights2.numpy(),
        'biases2':  biases2.numpy(),
        'final_weights': final_weights.numpy(),
        'final_biases':  final_biases.numpy()
    }
    
    # Save the dictionary as a .npz file
    np.savez(path, **model_dict)

#----------------------------------------------------------------------------------------
# Defining a function called get_weights_bias() that initializes weights and biases for 
# a neural network with two hidden layers. The function sets a fixed seed to ensure 
# reproducibility of the results and uses the normal distribution for initialization. 
# The parameters input_size, hidden_size1, and hidden_size2 determine the dimensions 
# of the weights and biases for the respective layers of the network:

def get_weights_bias(input_size, hidden_size1, hidden_size2):
    tf.random.set_seed(31)

    weights1 = tf.Variable(tf.random.normal([input_size, hidden_size1]))
    biases1 = tf.Variable(tf.random.normal([hidden_size1]))

    weights2 = tf.Variable(tf.random.normal([hidden_size1, hidden_size2]))
    biases2 = tf.Variable(tf.random.normal([hidden_size2]))

    final_weights = tf.Variable(tf.random.normal([hidden_size2, 1]))
    final_biases = tf.Variable(tf.random.normal([1]))

    return weights1, biases1, weights2, biases2, final_weights, final_biases

#----------------------------------------------------------------------------------------
# 
def neuron(x, weights, biases):
    z = tf.add(tf.matmul(x, weights), biases)
    return z

#----------------------------------------------------------------------------------------
# 
def execute():
    # Print library versions
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")  

    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    X = X[['age', 'chol', 'cp']]
    print("\n X.head(10)......")
    print(X.head(10))

    # Transform the 'cp' column into 4 columns
    df_cp = pd.get_dummies(X['cp'], prefix='cp')
    X = X.drop('cp', axis=1).join(df_cp)
    print("\n X.head(10)......")
    print(X.head(10))

    # Creating the target variable:
    target = heart_disease.data.targets
    print("\n target.head(10)......")    
    print(target.head(10))    
    target = (target > 0) * 1
    print("\n target.head(10)......")       
    print(target.head(10))    

    # Normalizing the data
    scaler = StandardScaler()
    X[['age', 'chol']] = scaler.fit_transform(X[['age', 'chol']])
    
    # Creating a constant input and a constant y
    input_data = tf.constant(X, dtype=tf.float32)
    print("\n input_data")       
    print(input_data)       
    y = tf.constant(target, dtype=tf.float32) 
    
    # Separating data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(input_data.numpy(), y.numpy(), test_size=0.2, stratify=y.numpy(), random_state=4321)

    print("\n X_train")       
    print(X_train)  
    
    # Normalizing the data
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    print("\n X_train Normalized")       
    print(X_train)  

    # Converting to TensorFlow tensors
    X_train = tf.constant(X_train, dtype=tf.float32)
    X_test  = tf.constant(X_test, dtype=tf.float32)
    y_train = tf.constant(y_train, dtype=tf.float32)
    y_test  = tf.constant(y_test, dtype=tf.float32)

    print(f"Shape of X_train: {tf.shape(X_train).numpy()}")
    print(f"Shape of X_test: {tf.shape(X_test).numpy()}")
    print(f"Shape of y_train: {tf.shape(y_train).numpy()}")
    print(f"Shape of y_test: {tf.shape(y_test).numpy()}")

    # Quantity features (inputs age, chol, cp1, cp2, cp3, cp4) 
    num_features = X_train.shape[1]

    optimizer = tf.optimizers.SGD(learning_rate=0.01)
    NUM_EPOCHS = 1500
    loss_calculator = tf.keras.losses.BinaryCrossentropy()

    losses = []
    accuracy_rates = []

    variables = get_weights_bias(num_features, 6, 4)
    weights1, biases1, weights2, biases2, final_weights, final_biases = variables

    for epoch in range(NUM_EPOCHS):
        with tf.GradientTape() as tape:
            # TRAIN DATASET
            train1 = tf.nn.relu(neuron(X_train, weights1, biases1))  # ReLU activation function added
            train2 = tf.nn.relu(neuron(train1, weights2, biases2))  # ReLU activation function added
            train3 = tf.sigmoid(neuron(train2, final_weights, final_biases))
            
            cost = loss_calculator(y_train, train3)
            losses.append(cost.numpy())
            
            # TEST DATASET
            test1 = tf.sigmoid(neuron(X_test, weights1, biases1))
            test2 = tf.sigmoid(neuron(test1, weights2, biases2))
            test3 = tf.sigmoid(neuron(test2, final_weights, final_biases))

            correct_predictions = np.mean(y_test.numpy() == ((test3.numpy() > 0.5) * 1))
            accuracy_rates.append(correct_predictions)
            
            gradients = tape.gradient(cost, variables)
            optimizer.apply_gradients(zip(gradients, variables))

    print(f'Lowest cost obtained with ReLU in hidden layers: {min(losses)}')
    print(f'Highest accuracy rate obtained with ReLU in hidden layers: {max(accuracy_rates)}') 

    plt.plot(losses)
    plt.plot(accuracy_rates)
    plt.title('Loss and Accuracy Rates per Epoch')
    plt.legend(['Training Loss', 'Test Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Rate')
    plt.ylim(0,1)
    plt.savefig('loss_and_accuracy_graph.png')
    
    save_model(weights1, biases1, weights2, biases2, final_weights, final_biases, 'saved_model.npz')


execute()
