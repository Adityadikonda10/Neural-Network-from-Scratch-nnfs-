
# Neural Network from Scratch (nnfs) 

Welcome to the Neural Network from Scratch (nnfs) repository! This project is inspired by the principles taught in sentdex's YouTube playlist and the book "Neural Network from Scratch." This repository dives into the fundamentals of building a neural network from scratch and understanding its components.




## Table of Contents

- Basic Principles of Neural Networks.
- Weights and Biases.
- Equations and Formulae Used.
    - Forward Propagation.
    - Activation Function (ReLU).
    - Softmax Activation.
    - Loss Calculation (Categorical Cross-Entropy).
    - Optimization.
    - Optimizer Algorithm Used.
- Deployment.
- Acknowledgements.



## Basic Principles of Neural Networks

Neural networks are computational models inspired by the human brain's neural structure. They consist of layers of interconnected nodes (neurons) that process information and learn patterns from data through iterative adjustments of weights and biases.

## Weights and Biases

**Weights**: In a neural network, weights $(W)$ determine the strength of connections between neurons in different layers. They are adjusted during training to minimize error and improve model performance.

**Biases**: Biases $(b)$ are additional parameters in neurons that allow them to better fit the data. They provide flexibility and help the model account for differences between the predicted outputs and the actual outputs.

## Equations and Formulae Used

### Forward Propagation
Forward propagation computes the output of the neural network from input data, passing through each layer using matrix multiplication and activation functions.

### Single Neuron output:

For single neuron, the output is calculated as the product of inputs and weights plus bias. The formula is:

$$ \text{neuron output} = \mathbf{x} \cdot \mathbf{w} + b $$

Where:
- $\mathbf{x}$ is the input to the neuron.
- $\mathbf{w}$ is the weight of the neuron.
- $\ b$ is the bias, a scalar value,


### Multiple Neurons Output:

For multiple neurons, the output is calculated as the summation of the product of inputs and weights plus biases. The formula is:

$$ \text{output} = \sum_{i=0}^{n} (\text{input}_i \cdot \text{weight}_i) + \text{biases} $$

Where:
- $\text{input}_i$ represents each input value.
- $\text{weight}_i$ represents each corresponding weight.
- $\text{biases}$ is the bias term added to the summation.
- \(n\) is the number of inputs.


### Activation Function (ReLU)

The Rectified Linear Unit (ReLU) activation function introduces non-linearity by outputting the input directly if positive, otherwise, it outputs zero:

$$ \text{ReLU}(x) = \max(0, x) $$

### Softmax Activation

Softmax activation is used in the output layer for multi-class classification. It converts raw scores into probabilities that sum to 1, allowing the model to predict class probabilities:

$$ \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}} $$

### Loss Calculation (Categorical Cross-Entropy)

Categorical Cross-Entropy loss measures the difference between predicted probabilities and actual target values for multi-class classification tasks:

$$ \text{Loss} = -\sum_{i} y_i \log(p_i) $$

where \( y_i \) is the target probability (one-hot encoded) and \( p_i \) is the predicted probability.

### Optimization

Optimization involves adjusting weights and biases to minimize the loss function during training. This process helps the model learn patterns and generalize to unseen data.

### Optimizer Algorithm Used

The Stochastic Gradient Descent (SGD) optimizer is implemented with momentum for faster convergence:

$$ \Delta \theta_{t+1} = \mu \cdot \Delta \theta_t - \eta \cdot \nabla_{\theta} L(\theta_t) $$

$$ \theta_{t+1} = \theta_t + \Delta \theta_{t+1} $$

where $\Delta$ is the learning rate, $\mu$  is the momentum coefficient, $\Delta \theta$ is the update, and $L(\theta)$ is the loss function.

## Deployment
To run the code and train the neural network, follow these steps:

1. #### Clone the Repository:
    ```sh
    git clone https://github.com/yourusername/nnfs.git
    cd nnfs
    ```
    <img width="1525" alt="Screenshot 2024-07-17 at 3 24 16 PM" src="https://github.com/user-attachments/assets/33772dc0-0990-47c0-8192-1ab242b24a9c">
2. #### Install Dependencies:
    Ensure you have Python installed. Then, install the required libraries using:

    ```sh
    pip install numpy nnfs
    ```
3. #### Run the Code:
    Execute the train.py script in the saved model directory to train the neural network:

    ```sh
    python saved_model/training_model.py
    ```
4. #### Training Results
    The training output will display the loss and accuracy at various epochs during the training process.
    <img width="1680" alt="Screenshot 2024-07-17 at 3 27 57 PM" src="https://github.com/user-attachments/assets/2817c41e-34c5-4932-928f-bcd4934abd00">


5. #### Generate test Dataset
    Execute the generate_dataset.py script in the saved model directory to test the neural network:
    ```sh
    python saved_model/generate_dataset.py
    ```
    <img width="1680" alt="Screenshot 2024-07-17 at 3 29 11 PM" src="https://github.com/user-attachments/assets/72cb6e9e-5b31-4dc1-ab7a-5084fe7d4d89">

6. #### Test Results
    The predictions will be displayed.
   ```sh
    python saved_model/testing_model.py
   ```
    <img width="1680" alt="Screenshot 2024-07-17 at 3 29 48 PM" src="https://github.com/user-attachments/assets/a587a9b5-a707-4c86-8ef3-0377f2b83c1a">

    
 
## Acknowledgements
This project is heavily inspired by the work of Harrison Kinsley (sentdex) and his YouTube playlist and book "Neural Network from Scratch." Thank you, Harrison Kinsley, for your comprehensive tutorials and explanations.
 
