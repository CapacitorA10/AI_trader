import torch
import torch.nn as nn

# Define the CNN
class CNN(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size)

    def forward(self, x):
        x = self.conv1(x)   # Conv1d : (N, C, L)
        return x # X: (N, C, L)

# Define the GRU
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        x = x.transpose(1,2)  # gru : (N, L, H)
        x, hidden = self.gru1(x)
        return x, hidden # X: (N, L, H)

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
x = torch.randn(batch_size, input_size, seq_length)

# Forward pass the input through the model
output, hidden = model(x)




######

# Load the data
data = # Load your data here

# Split the data into training and validation sets
train_data, val_data = # Split the data here

# Define the model hyperparameters
input_size = 1
num_filters = 8
kernel_size = 3
hidden_size = 8

# Create an instance of the model
model = Model(input_size=input_size, num_filters=num_filters, kernel_size=kernel_size, hidden_size=hidden_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
for epoch in range(num_epochs):
    # Training phase
    for data in train_data:
        # Extract the input and target
        x = data[0]
        y = data[1]

        # Forward pass
        output, _ = model(x)
        loss = criterion(output, y)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation phase
    with torch.no_grad():
        correct = 0
        total = 0
        for data in val_data:
            # Extract the input and target
            x = data[0]
            y = data[1]

            # Forward pass
            output, _ = model(x)

            # Compute the validation accuracy
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        val_accuracy = 100 * correct / total

    # Print the epoch-wise training and validation accuracies
    print("Epoch: {} | Train Loss: {:.4f} | Validation Accuracy: {:.2f}%".format(epoch+1, loss.item(), val_accuracy))

