import torch
import torch.nn as nn

# 3.2 task
# ---- Neural Network for Locomotion Control ----


class NeuralController(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralController, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)  # Hidden layer with 16 neurons
        self.fc2 = nn.Linear(16, output_size)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function
        x = self.fc2(x)  # Output layer
        return torch.tanh(x) * 100  # Outputs actions for the robot

# ---- Convert Weights to NumPy Arrays ----


def get_weights(model):
    """Extract weights from a PyTorch model as NumPy arrays."""
    return [p.detach().numpy() for p in model.parameters()]

# ---- Load Weights Back into a Model ----


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier initialization
        nn.init.constant_(m.bias, 0.1)  # Small bias
