import torch.nn as nn

# Define the CNN
class CNN(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)   # Conv1d : (N, C, L)
        x = self.relu(x)
        return x # X: (N, num_filters, L)

# Define the GRU
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        # x : (N, C, L) -> (N, L, C)
        x = x.permute(0,2,1)
        x, hidden = self.gru1(x)
        return x, hidden # X: (N, L, H)

# Define the overall model
class Model(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.cnn = CNN(input_size=input_size, num_filters=num_filters, kernel_size=kernel_size)
        self.gru = GRU(input_size=num_filters, hidden_size=hidden_size)
        self.fc = nn.Linear(18*hidden_size, output_size)
        self.output_size = output_size

    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.gru(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x
