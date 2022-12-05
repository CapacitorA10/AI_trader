import torch
import torch.nn as nn

# Define the CNN
class CNN(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size)

    def forward(self, x):
        x = self.conv1(x)
        return x

# Define the GRU
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size)

    def forward(self, x):
        x, hidden = self.gru1(x)
        return x, hidden

# Define the overall model
class Model(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, hidden_size):
        super(Model, self).__init__()
        self.cnn = CNN(input_size=input_size, num_filters=num_filters, kernel_size=kernel_size)
        self.gru = GRU(input_size=num_filters, hidden_size=hidden_size)

    def forward(self, x):
        x = self.cnn(x)
        x, hidden = self.gru(x)
        return x, hidden

# Create an instance of the model
input_size = 1
num_filters = 8
kernel_size = 3
hidden_size = 8
model = Model(input_size=input_size, num_filters=num_filters, kernel_size=kernel_size, hidden_size=hidden_size)

# Generate some dummy input data
seq_length = 10
batch_size = 2
x = torch.randn(seq_length, batch_size, input_size)

# Forward pass the input through the model
output, hidden = model(x)
