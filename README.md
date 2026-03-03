# Neural Network Implementation with Backpropagation

This project implements a neural network from scratch using NumPy. The code includes forward propagation, backward propagation, and gradient updates for training the network.

## Backpropagation Logic

The following code snippet demonstrates the backpropagation algorithm, which computes the gradients of the cost function with respect to the weights and biases of the network. These gradients are used to update the parameters during training.

### Code Snippet

```python
error = self.cost_derivative(activations[-1], y) * self.sigmoid_derivative(all_z[-1])
        # update last layers
        grad_b[-1] = error
        grad_w[-1] = np.dot(error, activations[-2].T) # a^(L-2) * error = partial derivative of cost wrt weight of last layer

        # Iterate backward through the layers (from second-to-last to input)
        # We iterate in reverse order of layers (from output towards input).
        # Layer 1 is the first hidden layer (weights[0], biases[0]).
        # Layer L-1 is the last hidden layer (weights[-2], biases[-2]).
        for layer_idx in reversed(range(self.num_layers - 1)):
            # skip last layer (we handled that above)
            if layer_idx == self.num_layers - 2: 
                continue 
            z = all_z[layer_idx]
            sd = self.sigmoid_derivative(z)
            # Error propagation: delta for current layer from delta of next layer
            error = np.dot(self.weights[layer_idx + 1].T, error) * sd

            # Gradients for current layer's biases and weights
```



###  Explanation
Output Layer Error:

The error at the output layer is computed as: [ \delta_L = \nabla_a C \odot \sigma'(z_L) ] where:
( \nabla_a C ) is the derivative of the cost function with respect to the output activations.
( \sigma'(z_L) ) is the derivative of the sigmoid function at the output layer.
Hidden Layer Error:

The error at each hidden layer is computed as: [ \delta_l = (W_{l+1}^T \delta_{l+1}) \odot \sigma'(z_l) ]
Gradient Calculation:

Gradients for biases: [ \frac{\partial C}{\partial b_l} = \delta_l ]
Gradients for weights: [ \frac{\partial C}{\partial W_l} = \delta_l \cdot a_{l-1}^T ]

### Purpose of the Code
The code snippet is part of the backpropagation algorithm, which is used to train the neural network by adjusting the weights and biases to minimize the cost function.

### How to Use
Clone the repository and ensure you have Python and NumPy installed.
Use the Network class to define your neural network architecture.
Train the network using the train_network function, which leverages the backpropagation algorithm.

### License
This project is licensed under the MIT License. Feel free to use and modify it as needed.
