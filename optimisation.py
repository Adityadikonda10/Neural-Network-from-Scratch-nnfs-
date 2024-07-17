import time
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons)) 

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class ActivationReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class ActivationSoftMax:
    def forward(self, inputs):
        self.exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.probablities = self.exp_values / np.sum(self.exp_values, axis=1, keepdims=True)
        self.output = self.probablities

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

class Loss:
    def calculate(self, output, y):
        self.sample_losses = self.forward(output, y)
        self.data_loss = np.mean(self.sample_losses)
        return self.data_loss

class LossCategoricalClassEntropy(Loss):
    def forward(self, y_pred, y_true):
        self.samples = len(y_pred)
        self.y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]
        self.correct_confidences = np.sum(self.y_pred_clipped * y_true, axis=1)
        self.negative_log_liklihoods = -np.log(self.correct_confidences)
        return self.negative_log_liklihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if y_true.ndim == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
        weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
        layer.weight_momentums = weight_updates
        bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
        layer.bias_momentums = bias_updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

# Training code
start_time = time.time()

X, y = spiral_data(100, 3)

dense1 = Layer_Dense(2, 3)
activation1 = ActivationReLU()

dense2 = Layer_Dense(3, 3)
activation2 = ActivationSoftMax()

loss_function = LossCategoricalClassEntropy()
optimizer = Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)
    
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)
    
    if not epoch % 100:
        print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learning_rate}')

    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

print(f"Finished in {round(time.time() - start_time, 2)}s")
