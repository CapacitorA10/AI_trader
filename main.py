##
import yfinance as yf

import torch
import torch.nn.init
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

device = 'cuda'
torch.cuda.is_available()
## data import
'''
stocks = ['SPY','TLT','GC=F']
start_date = '2003-01-01'
end_date = '2021-06-01'

stock_spy = yf.download('SPY',start_date,end_date)
stock_spy.to_csv('SPY'+'.csv',mode='w',header=True)

stock_tlt = yf.download('TLT',start_date,end_date)
stock_tlt.to_csv('TLT'+'.csv',mode='w',header=True)

stock_gold = yf.download('GC=F',start_date,end_date)
stock_gold.to_csv('GC=F'+'.csv',mode='w',header=True)

stock_spy["Adj Close"].plot()
stock_tlt["Adj Close"].plot()
stock_gold["Adj Close"].plot()
plt.show()
'''
## dataset 설정
class stockdataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # 30일씩 가져오도록 설정
        last_1m = torch.from_numpy(self.data.iloc[i:i+30].values[:,1:].astype(np.float)).float()
        after_1m = torch.from_numpy(self.data.iloc[i+30].values[1:].astype(np.float)).float()
        date = self.data.iloc[i+30].values[0]

        return date, last_1m, after_1m
spy = stockdataset('SPY.csv')
tlt = stockdataset('TLT.csv')
gold = stockdataset('GC=F.csv')

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
            torch.nn.ReLU()
        )
        self.CombiLayer = torch.nn.Sequential(
            torch.nn.Conv1d(192, 256, kernel_size=5, padding=2, bias=True),
            torch.nn.BatchNorm1d(256),                                    # 출력shape:([B, 64, 30])
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=True),  # 5일치 단위
            torch.nn.BatchNorm1d(256),  # 출력shape:([B, 128, 30])
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Linear(256*15, 3, bias=True) #open만 예측하기로...

    def forward(self, snp, bond, gold):
        snp = self.SnPLayer(snp); bond = self.BondLayer(bond); gold = self.GoldLayer(gold)
        out = torch.cat((snp, bond, gold), 1)
        out = self.CombiLayer(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc(out)
        return out

## for 3dim train
STOCKMODEL = comb_model1().to(device)
optimizer = torch.optim.Adam(STOCKMODEL.parameters(), lr=0.0001)
spy_loader = DataLoader(dataset=spy, batch_size=1, shuffle=False)

max_epoch = 30
k = 0
for epoch in range(max_epoch):
    loss = 0
    step = 0
    tlt_loader = iter(DataLoader(dataset=tlt, batch_size=1, shuffle=False))
    gold_loader = iter(DataLoader(dataset=gold, batch_size=1, shuffle=False))
    for date, inp_spy, ans_spy in spy_loader:
        # data load
        _, inp_tlt, ans_tlt = next(tlt_loader)
        gold_date, inp_gold, ans_gold = next(gold_loader)
        step += 1
        k += 1
        if gold_date[0] == '2017-12-29' : #2003 - 2017년까지만 학습
            print('load done')
            break

        # batch, feature, time step 순 정리
        inp_spy = inp_spy.permute(0,2,1).to(device)
        inp_tlt = inp_tlt.permute(0,2,1).to(device)
        inp_gold = inp_gold.permute(0,2,1).to(device)
        ans_spy = ans_spy[0, 0].to(device)
        ans_tlt = ans_tlt[0, 0].to(device)
        ans_gold = ans_gold[0, 0].to(device)

        # LEARNING START
        optimizer.zero_grad()
        pred = STOCKMODEL(inp_spy, inp_tlt, inp_gold)#.sum()
        cost = F.l1_loss(pred, torch.Tensor([[ans_spy, ans_tlt, ans_gold]]).to(device), reduction="none").to(device)
        cost = cost.sum()
        cost.backward()
        optimizer.step()

        # LOSS Calc
        loss += abs(cost)
        writer.add_scalar("Loss/train", abs(cost), k)

    loss /= step
    print(loss)
