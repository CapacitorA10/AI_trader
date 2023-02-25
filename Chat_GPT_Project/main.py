import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import dataPreprocessing as dpp
from model import TCN
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = 'cpu'

training_span = 128
cum_volatility = 5
batch_size = 8
test_size = 500
# tk10 = fdr.DataReader('KR10YT=RR')
hsi = fdr.DataReader('HSI')
kospi = fdr.DataReader('KS11')
krw_usd = fdr.DataReader('FRED:DEXKOUS')
t10y_2y = fdr.DataReader('FRED:T10Y2Y')
t10 = fdr.DataReader('FRED:DGS10')

## data 정규화
start_date = '1996-12-13'
hsi_1 = dpp.stock_diff(hsi, 1, start_date)
kospi_1 = dpp.stock_diff(kospi, 1, start_date)
hsi_5 = dpp.stock_diff(hsi, cum_volatility, start_date)
kospi_5 = dpp.stock_diff(kospi, cum_volatility, start_date)
t10_ = dpp.normalize(t10)[t10.index >= start_date]
krw_usd_ = (krw_usd / krw_usd.max())[krw_usd.index >= start_date]
t10y_2y_ = (t10y_2y / t10y_2y.max())[t10y_2y.index >= start_date]

## data 합치고 쪼개기
stock_all, split = dpp.merge_dataframes([kospi_1], [kospi_5], "drop")
stock_all_ = dpp.append_time_step(stock_all, training_span, cum_volatility, split)
stock_all_tensor = torch.Tensor(np.array(stock_all_))
train = stock_all_tensor[:-test_size, :, :]
test = stock_all_tensor[-test_size:, :, :]
split_date = stock_all.iloc[-test_size].name # 쪼갠 날짜 저장해두기
##
# Create a custom dataset class that wraps the tensor
class CustomDataset(data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index, :, :]

    def __len__(self):
        return self.tensor.size(0)


# Create an instance of the custom dataset
train_ = CustomDataset(train)
test_ = CustomDataset(test)
# Create a DataLoader with shuffle=True, but only shuffle the first dimension
train_dataloader = torch.utils.data.DataLoader(train_, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader =  torch.utils.data.DataLoader(test_, batch_size=1, shuffle=False)

## 신경망 수립

input_features = train.shape[-1]-1
hidden_features = input_features * 4
kernel_size = 3
num_layers = 7
dilation_rates = [2**i for i in range(num_layers)]

## Define the loss function and optimizer
model = TCN(input_features, hidden_features, kernel_size, dilation_rates, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train the model
num_epochs = 5
loss_ = 0
loss_cum = []
ticket = 0

# Create a figure and axis for plotting
fig, ax = plt.subplots()

# learning
for epoch in range(num_epochs):
    iter = 0
    model.train()
    for inputs in train_dataloader:
        # Forward pass
        inputs = inputs.to(device)
        targets = inputs[:,-1, 3].to(device)
        outputs = model(inputs[:, :-1, :-1])
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_ += loss


        # Print the loss every 10 iter
        iter += 1
        if (iter) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_:.3f}')
            loss_cum.append(loss_.cpu().detach())
            loss_ = 0

    loss_ = 0

    # 1 epoch ended
    # now, make a graph using test data
    with torch.no_grad():
        model.eval()
        inputs_cum = []
        targets_cum = []
        outputs_cum = []
        loss_cum = []
        for inputs in test_dataloader:
            inputs = inputs.to(device) ; inputs_cum.append(inputs.cpu().detach())
            targets = inputs[:, -1, 3].to(device) ; targets_cum.append(targets.cpu().detach())
            outputs = model(inputs[:, :-1, :-1]) ; outputs_cum.append(outputs.cpu().detach())
            loss = criterion(outputs, targets) ; loss_cum.append(loss.cpu().detach())

        # drawing
        # Plot the test data for this epoch
        inputs_all = torch.cat(inputs_cum, dim=0)
        targets_all = torch.cat(targets_cum, dim=0)
        outputs_all = torch.cat(outputs_cum, dim=0)
        loss_all = torch.mean(torch.stack(loss_cum))

        # Add legend label to the plot
        label = f'Epoch {epoch} - Loss: {loss_all:.4f}'

        # Plot the data
        ax.scatter(targets_all, outputs_all, label=label)

        # Set plot properties
        ax.set_xlabel('Test Targets')
        ax.set_ylabel('Test Outputs')
        ax.legend()
        plt.show()

# Set plot properties
ax.set_xlabel('Test Targets')
ax.set_ylabel('Test Outputs')
ax.legend()
plt.show()
##


for i in test_dataloader:
    pass
print(model(i[:, :-1, :-1].to(device)))

## test 날짜 따오기


kospi.loc[split_date:,'Close']

