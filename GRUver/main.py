##
import DataTools as DTs
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorboardX import SummaryWriter
from sklearn.preprocessing import StandardScaler, MinMaxScaler


writer = SummaryWriter()
device = 'cuda'
#input_t = int(input("input length? "))# 입력데이터 period 100 - 70 - 30 - 15 - 6 등...
#period = int(input("period term? "))#입력-출력간 기간

time_step = 20
bSize = 32   # 배치 사이즈
torch.cuda.is_available()
## Data import

stocks = DTs.data_import('2000-09-01', '2025-01-01') #item변수 전달 안하면, 기본 3개 나스닥 채권 금만 return

# 1day:  입력데이터
# 15day: 출력데이터
stocks_1day_change = DTs.pct_change_except_bond(stocks)
stocks_days_change =  DTs.pct_change_except_bond(stocks, 15)

##
#stock_1day_x = DTs.split(stocks_1day_change, '2000-01-01', '2019-06-30')
#stock_1day_y = DTs.split(stocks_15day_change, '2000-01-01', '2019-06-30')
#stock_5day_x = DTs.split(stocks_1day_change, '2019-07-01', '2025-01-01')
#stock_5day_y = DTs.split(stocks_15day_change, '2019-07-01', '2025-01-01')

# 입출력 / test train 으로 나누기
X_train = stocks_1day_change.loc['2000-01-01' : '2019-06-30']
Y_train = stocks_days_change.loc['2000-01-01' : '2019-06-30']
X_test = stocks_1day_change.loc['2019-07-01' : '2025-01-01']
Y_test = stocks_days_change.loc['2019-07-01' : '2025-01-01']

X_train = DTs.append_time_step(X_train, time_step)
Y_train = DTs.append_time_step(Y_train, time_step)
X_test =  DTs.append_time_step(X_test, time_step)
Y_test =  DTs.append_time_step(Y_test, time_step)


#print(f"Shape:\n Xtrain:{X_train.shape} Ytrain:{Y_train.shape}\n Xtest:{X_test.shape} Ytest:{Y_test.shape}")
## 데이터 후가공
#mm = MinMaxScaler()
#ss = StandardScaler()



## Model Define

class GRU(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(GRU, self).__init__()
        self.num_classes = num_classes
        self.num_layers  = num_layers
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.seq_length  = seq_length

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 64)
        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = (torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        output, (hn) = self.gru(x, (h_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out

## parameter define
num_epochs = 10
learning_rate = 0.0001

input_size  = 6
hidden_size = 12
num_layers  = 1
num_classes = 3 # 출력은 나스닥 채권 금 (+달러인덱스도?

model = GRU(num_classes,input_size,hidden_size,num_layers,time_step)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## START
for epoch in range(num_epochs):
    outputs = model.forward(X_train)
    optimizer.zero_grad()
    loss = criterion(outputs, Y_train)
    loss.backward()

    optimizer.step()
    print(f'Epoch : {epoch}, loss : {loss.item():1.5f}')
