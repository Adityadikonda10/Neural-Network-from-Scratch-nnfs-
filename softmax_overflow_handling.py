import time
import numpy as np
import nnfs

nnfs.init()
start_time = time.time()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

# exp_values = np.exp(layer_outputs)
"""Here expvalues have a tendency to overflow once we get large numbers. To overcome this we must substract the values of each output
   by the largest output value in the layer. this ensures that largest output value is 0 and everything lies below 0. So the range of 
   exponential value stays under 1, because exp of 0 is 1. """
exp_values = np.exp(layer_outputs - np.max(layer_outputs, axis=1, keepdims=True))

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(norm_values)


print(f"[Finished in {round(time.time() - start_time, 2)}s]")

