"""Calculating Loss with Categorical Cross-Entropy. formula is -ve sum of log of prediction * target output
   the target output vectors is a set of vector which is based on ONE HOT ENCODING where the lable index 
   among the target vector is 1 anr rest is 0."""
import time
start_time = time.time()
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
# import math

# softmax_output = [0.7, 0.2, 0.5]
# target_output = [1, 0, 0]

# loss = -(math.log(softmax_output[0])*target_output[0]+
#          math.log(softmax_output[1])*target_output[1]+
#          math.log(softmax_output[2])*target_output[2])
# print(loss)

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

class Loss:
    def calculate(self, output, y):
      self.sample_losses = self.forward(output, y)
      self.data_loss = np.mean(self.sample_losses)
      return self.data_loss
    
class LossCategoricalClassEntropy(Loss):
    def forward(self, y_pred, y_true):
        self.samoles = len(y_pred)
        self.y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]
            print(y_true.shape)
        if len(y_pred.shape) == 1:
            self.correct_confidences = self.y_pred_clipped[range(self.samples), y_true]
        elif len(y_pred.shape) == 2:
            self.correct_confidences = np.sum(self.y_pred_clipped*y_true, axis=1)
        self.negative_log_liklihoods = -np.log(self.correct_confidences)
        return self.negative_log_liklihoods

        

X, y = spiral_data(100, 3)

dense1 = Layer_Dense(2, 3)
activation1 = ActivationReLU()

dense2 = Layer_Dense(3, 3)
activation2 = ActivationSoftMax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_function = LossCategoricalClassEntropy()
loss = loss_function.calculate(activation2.output, y)
print("Loss: ", loss)


print(f"[Finisher in {round(time.time() - start_time, 2)}s]")