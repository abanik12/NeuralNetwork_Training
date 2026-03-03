import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(), # Converts image to tensor and scales to [0, 1]
    transforms.Normalize((0.1307,), (0.3081,)) # Standardize with MNIST's mean and std
])

# Download and load the training, testing data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Load data
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

class PyTorchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp_to_hidden = nn.Linear(3, 3) # inp to hidden layer
        self.relu = nn.ReLU() # activation function
        self.hidden_to_pred = nn.Linear(3, 3) # hidden to output layer

    # Takes input
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.inp_to_hidden(x)
        x = self.relu(x)
        x = self.hidden_to_pred(x)
        return x

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

# prediction
    self.inp_to_hidden = nn.Linear(784, 128)  # Input layer to hidden layer
    self.relu = nn.ReLU()  # Activation function
    self.hidden_to_pred = nn.Linear(128, 10)  # Hidden layer to output layer
    probs = torch.softmax(pred, dim=1)    
    self.inp_to_hidden = nn.Linear(784, 128)  # Input layer to hidden layer
    self.hidden_to_pred = nn.Linear(128, 10)  # Hidden layer to output layer
    prediction = torch.argmax(probs, dim=1) # take the highest probability (example : correct digit in our MNIST example)
    print("\nPredicted class probabilities:\n", probs) # probability of each class (example : 0 - 9 all digit probabilities)
    print("Predicted classes:", prediction) # predicted outputs
    print("True classes:", targets) # actual outputs