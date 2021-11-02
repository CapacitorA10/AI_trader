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


start_date = '2003-01-01'
end_date = '2021-10-31'
dateRange = pd.date_range(start=start_date, end=end_date)

stock_spy = yf.download('SPY',start_date,end_date)
#stock_spy.to_csv('SPY'+'.csv',mode='w',header=True)
stock_nsdq = yf.download('NQ=F',start_date,end_date)
#stock_nsdq.to_csv('NQ=F'+'.csv',mode='w',header=True)
stock_tlt = yf.download('TLT',start_date,end_date)
#stock_tlt.to_csv('TLT'+'.csv',mode='w',header=True)
stock_gold = yf.download('GC=F',start_date,end_date)
#stock_gold.to_csv('GC=F'+'.csv',mode='w',header=True)
stock_oil = yf.download('CL=F',start_date,end_date)
#stock_oil.to_csv('CL=F'+'.csv',mode='w',header=True)

stock_spy["Adj Close"].plot()
stock_tlt["Adj Close"].plot()
stock_gold["Adj Close"].plot()
stock_oil["Adj Close"].plot()
stock_nsdq["Adj Close"].plot()
plt.show()

## dataset 설정
class stockdataset(Dataset):
    def __init__(self):
        self.data_spy = stock_spy
        self.data_tlt = stock_nsdq
        self.data_oil = stock_tlt
        self.data_gold = stock_gold
        self.data_nsdq = stock_oil

    def __len__(self):
        return len(self.data_spy)

    def data_pre_process(self):
        def make_stationary(data):
            for i in range(1, len(data)):
                # 볼륨값을 제외한 나머지는 맨 뒤에서부터 앞에값의 차이로 구한다
                temp = data.iloc[-i]  # 맨뒤
                temp2 = data.iloc[-(i + 1)]  # 그 전날
                newtemp = temp[0:6] - temp2[0:5]
                newtemp['Volume'] = temp['Volume']  # 볼륨값 유지
                data.iloc[-i] = newtemp

        make_stationary(self.data_spy)
        make_stationary(self.data_tlt)
        make_stationary(self.data_oil)
        make_stationary(self.data_gold)
        make_stationary(self.data_nsdq)

    def __getitem__(self, i):

        i+=1

        #날짜기준은 가장 많은 spy
        spy_idx = self.data_spy.iloc[i:i+30].index
        spy = self.data_spy.loc[spy_idx].values[:,1:]


        last_1m = torch.from_numpy(self.data.iloc[i:i+30].values[:,1:].astype(np.float)).float()
        last_1m_p1 = torch.from_numpy(self.data.iloc[i-1:i + 29].values[:, 1:].astype(np.float)).float()
        diff_1m = last_1m_p1 - last_1m

        after_1m = torch.from_numpy(self.data.iloc[i+50].values[1:].astype(np.float)).float()
        after_1m_p1 = torch.from_numpy(self.data.iloc[i + 49].values[1:].astype(np.float)).float()
        after_diff = after_1m_p1 - after_1m
        # 기준일+30일까지는 학습, 그보다+20일이 이제 정답(약4주후 데이터를 예측)
        date = self.data.iloc[i+50].values[0]

        return date, diff_1m, after_diff


## data 전처리

dataset = stockdataset()
dataset.data_pre_process()

stock_spy["Adj Close"].plot()
stock_tlt["Adj Close"].plot()
stock_gold["Adj Close"].plot()
stock_oil["Adj Close"].plot()
stock_nsdq["Adj Close"].plot()
plt.show()

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

max_epoch = 10
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
        if gold_date[0] == '2021-01-29' : #2003 - 2017년까지만 학습
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

##

with torch.no_grad():
    # 예측
    gold_start_idx = np.where(np.asarray(gold.data.Date) == '2021-05-28')[0][0]
    bond_start_idx = np.where(np.asarray(tlt.data.Date) == '2021-05-28')[0][0]
    snp_start_idx  = np.where(np.asarray(spy.data.Date) == '2021-05-28')[0][0]

    gold_start_val = gold.data.iloc[gold_start_idx - 30: gold_start_idx].values[:, 1:].astype(np.float32)
    gold_start_val_p1 = gold.data.iloc[gold_start_idx - 31: gold_start_idx-1].values[:, 1:].astype(np.float32)
    inp_diff_gold = gold_start_val_p1 - gold_start_val

    bond_start_val = tlt.data.iloc[bond_start_idx - 30: bond_start_idx].values[:, 1:].astype(np.float32)
    bond_start_val_p1 = tlt.data.iloc[bond_start_idx - 31: bond_start_idx-1].values[:, 1:].astype(np.float32)
    inp_diff_bond = bond_start_val_p1 - bond_start_val

    snp_start_val = spy.data.iloc[snp_start_idx - 30: snp_start_idx].values[:, 1:].astype(np.float32)
    snp_start_val_p1 = spy.data.iloc[snp_start_idx - 31: snp_start_idx-1].values[:, 1:].astype(np.float32)
    inp_diff_snp = snp_start_val_p1 - snp_start_val


    gold_start_val = torch.unsqueeze(torch.from_numpy(inp_diff_gold), 0).permute(0, 2, 1).to(device)
    bond_start_val = torch.unsqueeze(torch.from_numpy(inp_diff_bond), 0).permute(0, 2, 1).to(device)
    snp_start_val = torch.unsqueeze(torch.from_numpy(inp_diff_snp), 0).permute(0, 2, 1).to(device)

    pred = STOCKMODEL(snp_start_val, bond_start_val, gold_start_val)

    print(pred)

    # 실제
    real_gold_p = spy.data.Open[np.where(np.asarray(gold.data.Date) == '2021-05-03')[0][0]]
    real_bond_p = tlt.data.Open[np.where(np.asarray(tlt.data.Date) == '2021-05-03')[0][0]]
    real_snp_p = spy.data.Open[np.where(np.asarray(spy.data.Date) == '2021-05-03')[0][0]]

    real_gold_f = spy.data.Open[np.where(np.asarray(gold.data.Date) == '2021-05-28')[0][0]]
    real_bond_f = tlt.data.Open[np.where(np.asarray(tlt.data.Date) == '2021-05-28')[0][0]]
    real_snp_f= spy.data.Open[np.where(np.asarray(spy.data.Date) == '2021-05-28')[0][0]]

    real = np.asarray([real_gold_f/real_gold_p-1, real_bond_f/real_bond_p-1,real_snp_f/real_snp_p-1]) *100

##

