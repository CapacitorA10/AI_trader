##
import DataTools as DTs
import torch
import torch.nn.init
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
device = 'cuda'
input_t = int(input("input length? "))# 입력데이터 period 100 - 70 - 30 - 15 - 6 등...
period = int(input("period term? "))#입력-출력간 기간
output_t = 1 # 출력데이터 길이
bSize = 64
torch.cuda.is_available()
## data import
stocks = DTs.data_import('2001-01-01', '2025-01-01')
periodDiff_stocks = DTs.data_pre_process_period(stocks, period=period)
DTs.data_pre_process_(stocks)
stock_train_x = DTs.split(stocks, '2001-01-01', '2019-06-30')
stock_train_y = DTs.split(periodDiff_stocks, '2001-01-01', '2019-06-30')
stock_test_x = DTs.split(stocks, '2019-07-01', '2025-01-01')
stock_test_y = DTs.split(periodDiff_stocks, '2019-07-01', '2025-01-01')
## dataset 설정
class stockdataset(Dataset):
    def __init__(self, stock_x, stock_y):
        self.x = {}
        self.y = {}
        for i in stock_x:
            self.x[i] = stock_x[i]
            self.y[i] = stock_y[i]

    def __len__(self):
        return len(self.x['spy']) - input_t - period + 1

    def __getitem__(self, i):

        targetDate = self.x['spy'].index[i + input_t + period - 1]
        inp = {}
        out = {}
        for j in self.x:
            inp[j] = DTs.pullout(self.x[j], i, i+input_t)
            outVal = self.y[j].loc[targetDate]
            out[j] = torch.from_numpy(outVal.values)

        date = outVal.name.strftime('%Y-%m-%d')

        return  date, inp, out

## data 전처리 및 show
trainData = stockdataset(stock_train_x, stock_train_y)
testData = stockdataset(stock_test_x, stock_test_y)

'''
trainData.x_spy.plot()
testData.y_spy.plot()
plt.show()
''' # 데이터 plot


## MODEL 3COMBI DEFINE
class comb_model1(torch.nn.Module):

    def __init__(self):
        super(comb_model1, self).__init__()
        self.SnPLayer = torch.nn.Sequential(
            # input = open/high/low/close/adjusted/volume
            torch.nn.Conv1d(6, 64, kernel_size=7, padding=3, bias=True), # input:6개, ouput:64, 7일치 단위
            torch.nn.BatchNorm1d(64),                                    # 출력shape:([B, 64, 30])
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, kernel_size=5, padding=2, bias=True),  # 5일치 단위
            torch.nn.BatchNorm1d(64),  # 출력shape:([B, 128, 30])
            #torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU()
        )
        self.BondLayer = torch.nn.Sequential(
            # input = open/high/low/close/adjusted/volume
            torch.nn.Conv1d(6, 64, kernel_size=7, padding=3, bias=True), # input:6개, ouput:64, 7일치 단위
            torch.nn.BatchNorm1d(64),                                    # 출력shape:([B, 64, 30])
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, kernel_size=5, padding=2, bias=True),  # 5일치 단위
            torch.nn.BatchNorm1d(64),  # 출력shape:([B, 128, 30])
            #torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU()
        )
        self.GoldLayer = torch.nn.Sequential(
            # input = open/high/low/close/adjusted/volume
            torch.nn.Conv1d(6, 64, kernel_size=7, padding=3, bias=True), # input:6개, ouput:64, 7일치 단위
            torch.nn.BatchNorm1d(64),                                    # 출력shape:([B, 64, 30])
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, kernel_size=5, padding=2, bias=True),  # 5일치 단위
            torch.nn.BatchNorm1d(64),  # 출력shape:([B, 128, 30])
            #torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU()
        )
        self.OilLayer = torch.nn.Sequential(
            # input = open/high/low/close/adjusted/volume
            torch.nn.Conv1d(6, 64, kernel_size=7, padding=3, bias=True),  # input:6개, ouput:64, 7일치 단위
            torch.nn.BatchNorm1d(64),  # 출력shape:([B, 64, 30])
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, kernel_size=5, padding=2, bias=True),  # 5일치 단위
            torch.nn.BatchNorm1d(64),  # 출력shape:([B, 128, 30])
            # torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU()
        )
        self.NasdaqLayer = torch.nn.Sequential(
            # input = open/high/low/close/adjusted/volume
            torch.nn.Conv1d(6, 64, kernel_size=7, padding=3, bias=True),  # input:6개, ouput:64, 7일치 단위
            torch.nn.BatchNorm1d(64),  # 출력shape:([B, 64, 30])
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, kernel_size=5, padding=2, bias=True),  # 5일치 단위
            torch.nn.BatchNorm1d(64),  # 출력shape:([B, 128, 30])
            # torch.nn.MaxPool1d(2, stride=2),
            torch.nn.ReLU()
        )
        self.CombiLayer = torch.nn.Sequential(
            torch.nn.Conv1d(320, 384, kernel_size=5, padding=2, bias=True),
            torch.nn.BatchNorm1d(384),
            torch.nn.ReLU(),
            torch.nn.Conv1d(384, 384, kernel_size=3, padding=1, bias=True),
            torch.nn.BatchNorm1d(384),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Linear(384*(input_t//2), output_t, bias=True) #input//2한 이유: 중간에 maxpool이 1번뿐이기 때문

    def forward(self, snp, bond, gold, oil, nasdaq):
        snp = self.SnPLayer(snp)
        bond = self.BondLayer(bond)
        gold = self.GoldLayer(gold)
        oil = self.OilLayer(oil)
        nasdaq = self.NasdaqLayer(nasdaq)

        out = torch.cat((snp, bond, gold, oil, nasdaq), 1)
        out = self.CombiLayer(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc(out)
        return out

## train ready
STOCKMODEL = comb_model1().to(device)
optimizer = torch.optim.Adam(STOCKMODEL.parameters(), lr=0.0001)
trainLoader = DataLoader(dataset=trainData, batch_size=bSize, shuffle=True, num_workers=0)
testLoader = DataLoader(dataset=testData, batch_size=1, shuffle=False, num_workers=0)

# tensorboard 그리기
_, sample_x, sample_y = next(iter(trainLoader))
writer.add_graph(STOCKMODEL,[sample_x['spy'].to(device)
                            ,sample_x['tlt'].to(device)
                            ,sample_x['gold'].to(device)
                            ,sample_x['oil'].to(device)
                            ,sample_x['nsdq'].to(device)])
max_epoch = 15
tb_step = 0
tb_step2 = 0
tb_avg_step2 = 0
## train start
for epoch in range(max_epoch):
    loss = 0
    step = 0

    for date, inVal, outVal in trainLoader:
        step += 1
        tb_step += 1

        # LEARNING START
        STOCKMODEL.train()
        optimizer.zero_grad()
        pred = STOCKMODEL(inVal['spy'].to(device),
                          inVal['tlt'].to(device),
                          inVal['gold'].to(device),
                          inVal['oil'].to(device),
                          inVal['nsdq'].to(device)).squeeze()
        # open/high/low/close/adjusted/volume
        ans_spy = outVal['spy'][:,0].to(device)
        cost = F.l1_loss(pred, ans_spy.squeeze(), reduction="none").to(device)
        cost.backward(cost)
        optimizer.step()
        # LOSS Calc
        loss += cost.sum() / (output_t*bSize)
        writer.add_scalar("Loss/train", cost.sum() / output_t, tb_step)

        # verification
        vLoss = 0
        step2 = 0
        if step % 2500 == 1:
            with torch.no_grad():
                STOCKMODEL.eval()
                for vDate, vInVal, vOutVal in testLoader:
                    vPred = STOCKMODEL(vInVal['spy'].to(device),
                                       vInVal['tlt'].to(device),
                                       vInVal['gold'].to(device),
                                       vInVal['oil'].to(device),
                                       vInVal['nsdq'].to(device)).squeeze()
                    vAns_spy = vOutVal['spy'][0][0].to(device)
                    vCost = F.l1_loss(vPred, vAns_spy.squeeze(), reduction="none").to(device)
                    writer.add_scalar("Loss/Test", vCost.sum() / output_t, tb_step2)
                    tb_step2 += 1
                    step2 += 1
                    vLoss += vCost / output_t
                print(f"verification done, Last date = {vDate}")
                print(f"mean Loss: {vLoss.sum() / step2}")
                print(f"each loss: {(vLoss*output_t) / step2}")
                writer.add_scalar("AvgLoss/Test", vLoss.sum()/step2, tb_avg_step2)
                tb_avg_step2 += 1


    loss /= step

    print(f"/////////////epoch{epoch} mean loss: {loss}///////////////")
## 평가용
STOCKMODEL.eval()

##

