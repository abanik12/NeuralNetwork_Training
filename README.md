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



# Neural Network Implementation with NumPy and PyTorch

This project demonstrates how to implement a neural network for the MNIST dataset using both NumPy and PyTorch. It includes forward propagation, backward propagation, and gradient updates for training the network.

---

## PyTorch Implementation

The PyTorch implementation of the neural network is provided in the `impl_tensor.py` file. This implementation uses PyTorch's `torch.nn` module to define the network layers and `torch.optim` for optimization.

### Key Features:
- **Data Loading**: Uses `torchvision.datasets` to download and preprocess the MNIST dataset.
- **Model Definition**: Defines a simple feedforward neural network with one hidden layer.
- **Training and Evaluation**: Includes training and evaluation loops for the model.

### Code Overview

#### 1. **Data Loading**
The MNIST dataset is loaded and preprocessed using PyTorch's `torchvision` library. The dataset is normalized to have a mean of `0.1307` and a standard deviation of `0.3081` (standard values for MNIST).

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image to tensor and scales to [0, 1]
    transforms.Normalize((0.1307,), (0.3081,))  # Standardize with MNIST's mean and std
])

# Download and load the training, testing data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Load data
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
```
#### 2. **Model Definition**

The neural network is defined using PyTorch's torch.nn.Module. It includes:

An input-to-hidden layer (inp_to_hidden).
A ReLU activation function.
A hidden-to-output layer (hidden_to_pred).

```python
import torch.nn as nn

class PyTorchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp_to_hidden = nn.Linear(784, 128)  # Input layer to hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.hidden_to_pred = nn.Linear(128, 10)  # Hidden layer to output layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.inp_to_hidden(x)
        x = self.relu(x)
        x = self.hidden_to_pred(x)
        return x
```

#### 3. **Training and Evaluation**

```python
import torch.optim as optim

# Initialize the model, loss function, and optimizer
model = PyTorchNet()
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the gradients
            output = model(data)  # Forward pass
            loss = criterion(output, target)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

# Evaluation loop
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)  # Get the index of the max log-probability
            correct += (predicted == target).sum().item()
            total += target.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Train and evaluate the model
train(model, train_loader, criterion, optimizer, epochs=5)
evaluate(model, test_loader)
```

### How to Run the PyTorch Implementation
1. Install Dependencies: Ensure you have PyTorch and torchvision installed. You can install them using pip:
 - pip install torch torchvision

2. Run the Script: Execute the impl_tensor.py file to train and evaluate the model:

- python impl_tensor.py

3. Expected Output:

The script will print the training loss for each epoch.
After training, it will evaluate the model on the test dataset and print the test accuracy.