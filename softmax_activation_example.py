import math
import time
import numpy as np
import nnfs

"""SoftMax Activation Function is a type of activation function which helps with dealing of negative values in the calculation
   The steps are as follows: (1) Exponitate the layer output values so that they are no more negative.
                             (2) Normalise the values [exp Nx / sum of exp N]"""

nnfs.init()

start_time = time.time()

layer_outputs = [4.8, 1.21, 2.385]

E = math.e

exp_values = np.exp(layer_outputs)
print(exp_values)
norm_values = exp_values/ sum(exp_values)
print(norm_values)
print(sum(norm_values))

# exp_values = [E**op for op in layer_outputs]
# print(exp_values)
# norm_values = [values/sum(exp_values) for values in exp_values]
# print(norm_values)
# print(sum(norm_values))

print(f"[Finished in {round(time.time() - start_time, 2)}s]")