##
import DataTools as DTs
from model import GRU
import torch.nn.init
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from sklearn.preprocessing import StandardScaler, MinMaxScaler

writer = SummaryWriter()
device = 'cuda'
torch.cuda.is_available()

time_step = 20
time_term = 15  # time step 막날로부터 xx일 후 예측
bSize = 2  # 배치 사이즈
learning_rate = 0.0001

stocks = DTs.data_import('2000-09-01', '2025-01-01')  # item변수 전달 안하면, 기본 3개 나스닥 채권 금만 return
stocks_1day_change = DTs.pct_change_except_bond(stocks)
stocks_days_change = DTs.pct_change_except_bond(stocks, time_term)
# 후가공?
# mm = MinMaxScaler()
# ss = StandardScaler()

## data split / XY merge
X_train = DTs.append_time_step(stocks_1day_change.loc['2000-01-01': '2019-06-30'],
                               stocks_days_change.loc['2000-01-01': '2019-06-30'],
                               time_step, 1)

X_test = DTs.append_time_step(stocks_1day_change.loc['2019-07-01': '2025-01-01'],
                              stocks_days_change.loc['2019-07-01': '2025-01-01'],
                              time_step, 1)

X_train = torch.FloatTensor(np.asarray(X_train)).to(device)
X_test = torch.FloatTensor(np.asarray(X_test)).to(device)

train_loader = torch.utils.data.DataLoader(dataset=X_train, batch_size=bSize, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=X_test, batch_size=1, shuffle=False)
# print(f"Shape:\n Xtrain:{X_train.shape} Ytrain:{Y_train.shape}\n Xtest:{X_test.shape} Ytest:{Y_test.shape}")


## START LEARN


input_size = X_train.shape[-1]
gru_out_size = 12  # (==hidden size)
num_layers = 1
num_classes = X_train.shape[-1] // 2  # 출력은 나스닥 채권 금 close.. (+달러인덱스도?

MODEL = GRU(num_classes, input_size, gru_out_size, num_layers, time_step).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(MODEL.parameters(), lr=learning_rate)

writer.add_graph(MODEL, X_train)
writer.add_graph(MODEL, X_test)

loss_graph = []  # 그래프 그릴 목적인 loss.
n = len(train_loader)
num_epochs = 20
##
step = 0
v_step = 0

for epoch in range(num_epochs):
    avg_train = 0
    for data in train_loader:

        MODEL.train()
        optimizer.zero_grad()

        out = MODEL(data[:, :-1, :])
        loss = criterion(out, data[:, -1, 0:3])
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss.sum() / bSize, step)
        step += 1
        avg_train += loss
        # verification

        if step % 2000 == 0:
            avg_test = 0
            with torch.no_grad():
                MODEL.eval()
                for test_data in test_loader:
                    out = MODEL(test_data[:, :-1, :])
                    loss = criterion(out, test_data[:, -1, 0:3])
                    avg_test += loss
                    writer.add_scalar("Loss/test", loss.sum() / bSize, v_step)
                    writer.add_scalar("REAL/GOLD", test_data.squeeze()[-1, 0], v_step)
                    writer.add_scalar("REAL/NSDQ", test_data.squeeze()[-1, 1], v_step)
                    writer.add_scalar("REAL/TRES", test_data.squeeze()[-1, 2], v_step)
                    writer.add_scalar("PREDICTED/GOLD", out.squeeze()[0], v_step)
                    writer.add_scalar("PREDICTED/NSDQ", out.squeeze()[1], v_step)
                    writer.add_scalar("PREDICTED/TRES", out.squeeze()[2], v_step)
                    v_step += 1
                print(f"TEST:  epoch/step:{epoch}/{step},  avg loss:{avg_test / X_test.shape[0]}")

    print(f"TRAIN: epoch:{epoch},        avg loss:{avg_train / X_train.shape[0]}\n")
## 최종 모델로 오늘로부터 3주 보기
experiments = torch.FloatTensor(np.asarray(stocks_1day_change.iloc[-20:, :])).unsqueeze(0)
predicted = MODEL(experiments.to(device))
