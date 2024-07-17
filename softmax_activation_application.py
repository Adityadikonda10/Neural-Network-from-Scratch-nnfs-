import time
start_time = time.time()
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons)) 

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class ActivationSoftMax:
    def forward(self, inputs):
        self.exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.probablities = self.exp_values / np.sum(self.exp_values, axis=1, keepdims=True)
        self.output = self.probablities
        

X, y = spiral_data(100, 3)

dense1 = Layer_Dense(2, 3)
activation1 = ActivationReLU()

dense2 = Layer_Dense(3, 3)
activation2 = ActivationSoftMax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])
print(f"[Finisher in {round(time.time() - start_time, 2)}s]")