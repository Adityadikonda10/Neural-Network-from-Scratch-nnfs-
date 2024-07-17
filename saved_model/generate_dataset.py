import numpy as np
import matplotlib.pyplot as plt
import os

# Function to generate spiral dataset
def generate_spiral_dataset(samples=100, classes=3):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2  # theta
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

# Generate new dataset
X, y = generate_spiral_dataset(samples=100, classes=3)

# Plot and save dataset
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Spiral Data')

# Ensure directory for saving datasets exists
if not os.path.exists('spiral_datasets_test'):
    os.makedirs('spiral_datasets_test')

# Find next available filename
dataset_id = 1
while True:
    filename = f'spiral_datasets_test/spiral_dataset_test_{dataset_id}.npz'
    if not os.path.exists(filename):
        break
    dataset_id += 1

# Save dataset
np.savez(filename, X=X, y=y)
print(f"Dataset saved as {filename}")

plt.show()
