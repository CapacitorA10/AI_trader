##
import ai_trader.DataTools as DTs
from ai_trader.model import GRU
import torch.nn.init
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
writer = SummaryWriter()
device = 'cuda'
torch.cuda.is_available()


time_step = 20
time_term = 15 #time step 막날로부터 xx일 후 예측
bSize = 1   # 배치 사이즈
learning_rate = 0.0001

stocks = DTs.data_import('2000-09-01', '2025-01-01') #item변수 전달 안하면, 기본 3개 나스닥 채권 금만 return
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


## START LEARN


input_size  = X_train.shape[-1]
gru_out_size = 12
num_layers  = 1
num_classes = X_train.shape[-1]//2 # 출력은 나스닥 채권 금 close.. (+달러인덱스도?

MODEL = GRU(num_classes, input_size, gru_out_size, num_layers, time_step).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(MODEL.parameters(), lr=learning_rate)

writer.add_graph(MODEL, X_train)
writer.add_graph(MODEL, X_test)

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
