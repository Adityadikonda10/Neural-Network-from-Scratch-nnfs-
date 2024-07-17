import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

with open('model.pkl', 'rb') as f:
    model_parameters = pickle.load(f)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = model_parameters["dense1_weights"] if n_inputs == 2 else model_parameters["dense2_weights"]
        self.biases = model_parameters["dense1_biases"] if n_inputs == 2 else model_parameters["dense2_biases"]

    def forward(self, inputs):
        self.inputs = inputs  # Store inputs for use in backward pass
        self.output = np.dot(inputs, self.weights) + self.biases

def load_and_predict_datasets():
    dataset_files = [f for f in os.listdir('spiral_datasets_test') if f.endswith('.npz')]

    for dataset_file in dataset_files:
        loaded_data = np.load(os.path.join('spiral_datasets_test', dataset_file))
        X_loaded, y_loaded = loaded_data['X'], loaded_data['y']

        dense1 = Layer_Dense(2, 64)
        dense2 = Layer_Dense(64, 3)

        dense1.forward(X_loaded)
        dense2.forward(dense1.output)
        predictions = np.argmax(dense2.output, axis=1)

        plt.figure(figsize=(8, 8))
        plt.scatter(X_loaded[:, 0], X_loaded[:, 1], c=y_loaded, cmap='brg', label='Dataset')
        plt.scatter(X_loaded[:, 0], X_loaded[:, 1], c=predictions, marker='x', cmap='Set1', s=100, linewidths=3, edgecolors='black', label='Predictions')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Loaded Spiral Data and Predictions ({dataset_file})')
        plt.legend()
        plt.show()

load_and_predict_datasets()
