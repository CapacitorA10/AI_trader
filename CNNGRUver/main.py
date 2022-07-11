##
import ai_trader.DataTools as DTs
from ai_trader.CnnGruModel import CNNGRU
import torch.nn.init
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

writer = SummaryWriter()
device = 'cpu'
torch.cuda.is_available()

time_step = 20
time_term = 5  # time step 막날로부터 xx일 후 예측
"""if time_term > time_step:
    print("ERR: time step MUST bigger than time term")
    exit()"""
bSize = 2  # 배치 사이즈
learning_rate = 0.0001
num_epochs = 100

#stocks = DTs.data_import('2000-09-01', '2025-01-01')  # item변수 전달 안하면, 기본 3개 나스닥 채권 금만 return
stocks = pd.read_csv('data.csv',header=[0,1],index_col=0)
stocks_1day_change = DTs.pct_change_except_bond(stocks)
stocks_days_change = DTs.pct_change_except_bond(stocks, time_term)
# 후가공-표준화
stocks_1day_change = (stocks_1day_change-stocks_1day_change.mean())/stocks_1day_change.std()
## data split / XY merge
split_date = '2020-12-30'
X_train = DTs.append_time_step(stocks_1day_change.loc['2000-01-01': split_date],
                               stocks_days_change.loc['2000-01-01': split_date],
                               time_step, time_term)

X_test = DTs.append_time_step(stocks_1day_change.loc[split_date: '2025-01-01'],
                              stocks_days_change.loc[split_date: '2025-01-01'],
                              time_step, time_term)

X_train = torch.FloatTensor(np.asarray(X_train)).transpose(1,2).to(device)
X_test = torch.FloatTensor(np.asarray(X_test)).transpose(1,2).to(device)

train_loader = torch.utils.data.DataLoader(dataset=X_train, batch_size=bSize, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=X_test, batch_size=1, shuffle=False)
# print(f"Shape:\n Xtrain:{X_train.shape} Ytrain:{Y_train.shape}\n Xtest:{X_test.shape} Ytest:{Y_test.shape}")
# test가 시작하는 때의 value(그래프 그리기용)
test_start_idx = stocks.index.get_loc(split_date)
test_real_start = torch.FloatTensor(stocks.iloc[test_start_idx - time_term,0:3]).to(device)

## START LEARN


input_size = X_train.shape[-2]
gru_out_size = 8  # (==hidden size)
num_layers = 1
num_classes = X_train.shape[-2] // 2  # 출력은 나스닥 채권 금 close.. (+달러인덱스도?

MODEL = CNNGRU(num_classes, input_size, gru_out_size, num_layers, time_step).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(MODEL.parameters(), lr=learning_rate)

#writer.add_graph(MODEL, X_train)
#writer.add_graph(MODEL, X_test)

##
step = 0
v_step = 0

for epoch in range(num_epochs):
    avg_train = 0
    for data in train_loader:

        MODEL.train()
        optimizer.zero_grad()

        out = MODEL(data[:, :, :-1])
        loss = criterion(out, data[:,0:3,-1])
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss.sum() / bSize, step)
        step += 1
        avg_train += loss

        # verification
        if step % 2000 == 0:
            avg_test = 0
            # 변화량 ->실제값 그래프 그리기 위해 값 추출
            gold = test_real_start.clone()[0]
            nasdaq = test_real_start.clone()[1]
            i_step = 0
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
                    gold *= (100 + out.squeeze()[0]) * 0.01
                    nasdaq *= (100 + out.squeeze()[1]) * 0.01
                    date_now = test_start_idx + time_term + time_step + i_step
                    writer.add_scalars("static-GOLD",{"REAL": stocks.iat[date_now, 0],
                                                      "PRED": gold} , v_step)
                    writer.add_scalars("static-NSDQ", {"REAL": stocks.iat[date_now, 1],
                                                       "PRED": nasdaq}, v_step)
                    writer.add_scalars("static-TRES", {"REAL": stocks.iat[date_now, 2],
                                                       "PRED": out.squeeze()[2]}, v_step)
                    i_step += 1
                    v_step += 1
                print(f"TEST:  epoch/step: {epoch}/{step},  avg loss: {avg_test / X_test.shape[0]}")
        torch.cuda.empty_cache()
    print(f"TRAIN: epoch: {epoch},  avg loss: {avg_train / X_train.shape[0]}\n")
## 최종 모델로 오늘로부터 3주 보기
experiments = torch.FloatTensor(np.asarray(stocks_1day_change.iloc[-time_step:, :])).unsqueeze(0)
predicted = MODEL(experiments.to(device))
print(predicted)
##
##
