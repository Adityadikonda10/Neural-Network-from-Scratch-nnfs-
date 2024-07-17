import numpy as np

inputs = [ [1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, -0.8],
          [-11.5, 2.7, 3.3, -0.8]]

weights1= [[0.2, 0.8, -0.5, 1.0],
          [0.5, -0.91, 0.26, -0.5],
          [-0.26, -0.27, 0.17, 0.87]]

weights2= [[0.4, 0.91, -0.15, 0.02],
          [-0.5, 0.83, -0.72, 0.19],
          [-0.56, -0.97, -0.71, 0.87]]
biases1 = [2, 3, 0.5]

biases2 = [0.2, 5, 0.8]

layer1 = np.dot(inputs, np.array(weights1).T) + biases1
# layer2 = np.dot(layer1, np.array(weights2).T) + biases2
print(layer1.shape())