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
time_term = 15 #time step 막날로부터 xx일 후 예측
bSize = 1   # 배치 사이즈
torch.cuda.is_available()


stocks = DTs.data_import('2000-09-01', '2025-01-01') #item변수 전달 안하면, 기본 3개 나스닥 채권 금만 return

# 1day:  입력데이터
# 15day: 출력데이터
stocks_1day_change = DTs.pct_change_except_bond(stocks)
stocks_days_change =  DTs.pct_change_except_bond(stocks, time_term)
#후가공?
#mm = MinMaxScaler()
#ss = StandardScaler()

## data split / XY merge
X_train = DTs.append_time_step(stocks_1day_change.loc['2000-01-01' : '2019-06-30'],
                     stocks_days_change.loc['2000-01-01' : '2019-06-30'],
                     time_step, 1)

X_test = DTs.append_time_step(stocks_1day_change.loc['2019-07-01' : '2025-01-01'],
                     stocks_days_change.loc['2019-07-01' : '2025-01-01'],
                     time_step, 1)

X_train = torch.FloatTensor(np.asarray(X_train)).to(device)
X_test = torch.FloatTensor(np.asarray(X_test)).to(device)

train_loader = torch.utils.data.DataLoader(dataset=X_train, batch_size=bSize, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=X_test, batch_size=bSize, shuffle=True)
#print(f"Shape:\n Xtrain:{X_train.shape} Ytrain:{Y_train.shape}\n Xtest:{X_test.shape} Ytest:{Y_test.shape}")


## Model Define

class GRU(nn.Module):
    def __init__(self, num_classes, input_size, gru_out_size, num_layers, seq_length):
        super(GRU, self).__init__()
        self.num_classes = num_classes
        self.num_layers  = num_layers
        self.input_size  = input_size
        self.hidden_size = gru_out_size
        self.seq_length  = seq_length

        self.gru = nn.GRU(input_size=input_size, hidden_size=gru_out_size,
                          num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(gru_out_size, 64)
        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        h_0 = (torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        output, (hn) = self.gru(x, (h_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.tanh(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out

## parameter define

learning_rate = 0.0001

input_size  = X_train.shape[-1]
gru_out_size = 12
num_layers  = 1
num_classes = X_train.shape[-1]//2 # 출력은 나스닥 채권 금 close.. (+달러인덱스도?

MODEL = GRU(num_classes,input_size,gru_out_size,num_layers,time_step).to(device)
MODEL = MODEL.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(MODEL.parameters(), lr=learning_rate)

## START
writer.add_graph(MODEL,X_train)
writer.add_graph(MODEL,X_test)

loss_graph = [] # 그래프 그릴 목적인 loss.
n = len(train_loader)
step = 0
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0

    for data in train_loader:

        out = MODEL(data[:,:-1,:])  # 모델에 넣고,
        loss = criterion(out, data[:,-1,0:3]) # output 가지고 loss 구하고,
        writer.add_scalar("Loss/train", loss.sum() / (bSize), step)
        optimizer.zero_grad() #
        loss.backward() # loss가 최소가 되게하는
        optimizer.step() # 가중치 업데이트 해주고,
        running_loss += loss.item() # 한 배치의 loss 더해주고,
        step+=1

    loss_graph.append(running_loss / n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
    print('[epoch: %d] loss: %.4f'%(epoch, running_loss))

##

