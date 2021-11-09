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
end_date = '2021-11-09'
dateRange = pd.date_range(start=start_date, end=end_date)

stock_spy = yf.download('SPY',start_date,end_date)
stock_nsdq = yf.download('NQ=F',start_date,end_date)
stock_tlt = yf.download('TLT',start_date,end_date)
stock_gold = yf.download('GC=F',start_date,end_date)
stock_oil = yf.download('CL=F',start_date,end_date)

stock_spy["Adj Close"].plot()
stock_tlt["Adj Close"].plot()
stock_gold["Adj Close"].plot()
stock_oil["Adj Close"].plot()
stock_nsdq["Adj Close"].plot()
plt.show()

## dataset 설정

input_t = 30 # 입력데이터 period
output_t = 10 # 출력데이터 period

class stockdataset(Dataset):
    def __init__(self):
        self.data_spy = stock_spy
        self.data_tlt = stock_tlt
        self.data_oil = stock_oil
        self.data_gold = stock_gold
        self.data_nsdq = stock_nsdq

    def __len__(self):
        return len(self.data_spy)

    def data_pre_process(self):
        def make_stationary(data):
            for i in range(1, len(data)):
                # 볼륨값을 제외한 나머지는 맨 뒤에서부터 앞에값의 차이로 구한다
                temp = data.iloc[-i]  # 맨뒤
                temp2 = data.iloc[-(i + 1)]  # 그 전날
                newtemp = ((temp[0:6] / temp2[0:5]) - 1)*100 # 당일/전날 하여 전날대비상승률 계산, 1을 빼서 정확한 %계산
                newtemp['Volume'] = temp['Volume']  # 볼륨값 유지
                data.iloc[-i] = newtemp
            # Volume 평준화 시행
            data['Volume'] = data['Volume'] / data['Volume'].max()



        make_stationary(self.data_spy)
        make_stationary(self.data_tlt)
        #make_stationary(self.data_oil) #oil 선물은 그냥 사용? -> 2020 4월에 음전한 경력이 있어 상승,하락률 계산도 어려움
        make_stationary(self.data_gold)
        make_stationary(self.data_nsdq)
        # oil은 Volume값 최적화만 수행
        self.data_oil['Volume'] = self.data_oil['Volume'] / self.data_oil['Volume'].max()

        # 데이터 숫자 맞추기 #
        # reindex를 사용한다? -> 가장 많은 놈 기준으로.. 현재 spy, tlt가 최다
        # reindex에서 gold, nsdq은 변화율 기준이니 값 0을 입력해주면 됨
        # 하지만 oil은 절대값 기준이므로 이전값을 대입하면 됨
        self.data_gold = self.data_gold.reindex(self.data_spy.index, fill_value=0)
        self.data_nsdq = self.data_nsdq.reindex(self.data_spy.index, fill_value=0)
        self.data_oil = self.data_oil.reindex(self.data_spy.index, method='ffill')

    def __getitem__(self, i):

        i+=1
        # pytorch가 이용 가능한 형태로 데이터 추출(index 및 기간 설정)
        def pullout(data,idx,period):
            # permute 사용해서 색인 축(open,close등)과 day축을 교환
            return (torch.from_numpy(data.iloc[idx:idx+period].values.astype(np.float64)).float()).permute(1,0)


        input_spy = pullout(self.data_spy, i, input_t)
        input_tlt = pullout(self.data_tlt, i, input_t)
        input_oil = pullout(self.data_oil, i, input_t)
        input_gold = pullout(self.data_gold, i, input_t)
        input_nsdq = pullout(self.data_nsdq, i, input_t)
        # 관리가 용이하도록 dictionary 형태로 모으기
        input_dic = {'spy':input_spy, 'tlt':input_tlt, 'oil':input_oil, 'gold':input_gold, 'nsdq':input_nsdq}

        output_spy = pullout(self.data_spy, i+input_t, output_t) # 출력값은 입력한 날 바로 다음날부터 가져옴!
        output_tlt = pullout(self.data_tlt, i+input_t, output_t)
        output_oil = pullout(self.data_oil, i+input_t, output_t)
        output_gold = pullout(self.data_gold, i+input_t, output_t)
        output_nsdq = pullout(self.data_nsdq, i+input_t, output_t)
        # 관리가 용이하도록 dictionary 형태로 모으기
        output_dic = {'spy': output_spy, 'tlt': output_tlt, 'oil': output_oil, 'gold': output_gold, 'nsdq': output_nsdq}

        date = self.data_spy.index[i+output_t].strftime('%Y-%m-%d')

        return  date, input_dic, output_dic


## data 전처리

dataset = stockdataset()
dataset.data_pre_process()

dataset.data_spy["Adj Close"][1:].plot()
dataset.data_tlt["Adj Close"][1:].plot()
dataset.data_gold["Adj Close"][1:].plot()
dataset.data_oil["Adj Close"][1:].plot()
dataset.data_nsdq["Adj Close"][1:].plot()
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
            torch.nn.BatchNorm1d(384),                                    # 출력shape:([B, 64, 30])
            torch.nn.ReLU(),
            torch.nn.Conv1d(384, 384, kernel_size=3, padding=1, bias=True),  # 5일치 단위
            torch.nn.BatchNorm1d(384),  # 출력shape:([B, 128, 30])
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

## for 3dim train
STOCKMODEL = comb_model1().to(device)
optimizer = torch.optim.Adam(STOCKMODEL.parameters(), lr=0.0001)
loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

max_epoch = 10
k = 0
for epoch in range(max_epoch):
    loss = 0
    step = 0
    for date, inVal, outVal in loader:
        step += 1
        k += 1
        if date[0] == '2019-07-31': #2003 - 2019년 여름까지만 학습
            print('load done')
            break
        # input = open/high/low/close/adjusted/volume
        # batch, feature, time step 순 정리
        inp_spy = inVal['spy'].to(device)
        inp_tlt = inVal['tlt'].to(device)
        inp_gold = inVal['gold'].to(device)
        inp_oil = inVal['oil'].to(device)
        inp_nsdq = inVal['nsdq'].to(device)

        # open 만 예상
        ans_spy = outVal['spy'][0][0].to(device)

        # LEARNING START
        optimizer.zero_grad()
        pred = STOCKMODEL(inp_spy, inp_tlt, inp_gold, inp_oil, inp_nsdq).squeeze()
        cost = F.l1_loss(pred, ans_spy, reduction="none").to(device)
        #cost = cost.sum()
        cost.backward(cost)
        optimizer.step()

        # LOSS Calc
        loss += abs(cost).sum()
        writer.add_scalar("Loss/train", abs(cost).sum(), k)

    loss /= step
    print(loss)
#######################################################################################################################
################################################## 여기서부터 검증 ######################################################
## data import


start_date = '2019-08-01'
end_date = '2021-11-09'
dateRange = pd.date_range(start=start_date, end=end_date)

vstock_spy = yf.download('SPY',start_date,end_date)
vstock_nsdq = yf.download('NQ=F',start_date,end_date)
vstock_tlt = yf.download('TLT',start_date,end_date)
vstock_gold = yf.download('GC=F',start_date,end_date)
vstock_oil = yf.download('CL=F',start_date,end_date)

vstock_spy["Adj Close"].plot()
vstock_tlt["Adj Close"].plot()
vstock_gold["Adj Close"].plot()
vstock_oil["Adj Close"].plot()
vstock_nsdq["Adj Close"].plot()
plt.show()

## dataset 설정 및 전처리

input_t = 30 # 입력데이터 period
output_t = 10 # 출력데이터 period

class vStockDataset(Dataset):
    def __init__(self):
        self.data_spy = vstock_spy
        self.data_tlt = vstock_tlt
        self.data_oil = vstock_oil
        self.data_gold = vstock_gold
        self.data_nsdq = vstock_nsdq

    def __len__(self):
        return len(self.data_spy)

    def data_pre_process(self):
        def make_stationary(data):
            for i in range(1, len(data)):
                # 볼륨값을 제외한 나머지는 맨 뒤에서부터 앞에값의 차이로 구한다
                temp = data.iloc[-i]  # 맨뒤
                temp2 = data.iloc[-(i + 1)]  # 그 전날
                newtemp = ((temp[0:6] / temp2[0:5]) - 1)*100 # 당일/전날 하여 전날대비상승률 계산, 1을 빼서 정확한 %계산
                newtemp['Volume'] = temp['Volume']  # 볼륨값 유지
                data.iloc[-i] = newtemp
            # Volume 평준화 시행
            data['Volume'] = data['Volume'] / data['Volume'].max()

        make_stationary(self.data_spy)
        make_stationary(self.data_tlt)
        #make_stationary(self.data_oil) #oil 선물은 그냥 사용? -> 2020 4월에 음전한 경력이 있어 상승,하락률 계산도 어려움
        make_stationary(self.data_gold)
        make_stationary(self.data_nsdq)
        # oil은 Volume값 최적화만 수행
        self.data_oil['Volume'] = self.data_oil['Volume'] / self.data_oil['Volume'].max()

        # 데이터 숫자 맞추기 #
        # reindex를 사용한다? -> 가장 많은 놈 기준으로.. 현재 spy, tlt가 최다
        # reindex에서 gold, nsdq은 변화율 기준이니 값 0을 입력해주면 됨
        # 하지만 oil은 절대값 기준이므로 이전값을 대입하면 됨
        self.data_gold = self.data_gold.reindex(self.data_spy.index, fill_value=0)
        self.data_nsdq = self.data_nsdq.reindex(self.data_spy.index, fill_value=0)
        self.data_oil = self.data_oil.reindex(self.data_spy.index, method='ffill')

    def __getitem__(self, i):

        i+=1
        # pytorch가 이용 가능한 형태로 데이터 추출(index 및 기간 설정)
        def pullout(data,idx,period):
            # permute 사용해서 색인 축(open,close등)과 day축을 교환
            return (torch.from_numpy(data.iloc[idx:idx+period].values.astype(np.float64)).float()).permute(1,0)


        input_spy = pullout(self.data_spy, i, input_t)
        input_tlt = pullout(self.data_tlt, i, input_t)
        input_oil = pullout(self.data_oil, i, input_t)
        input_gold = pullout(self.data_gold, i, input_t)
        input_nsdq = pullout(self.data_nsdq, i, input_t)
        # 관리가 용이하도록 dictionary 형태로 모으기
        input_dic = {'spy':input_spy, 'tlt':input_tlt, 'oil':input_oil, 'gold':input_gold, 'nsdq':input_nsdq}

        output_spy = pullout(self.data_spy, i+input_t, output_t) # 출력값은 입력한 날 바로 다음날부터 가져옴!
        output_tlt = pullout(self.data_tlt, i+input_t, output_t)
        output_oil = pullout(self.data_oil, i+input_t, output_t)
        output_gold = pullout(self.data_gold, i+input_t, output_t)
        output_nsdq = pullout(self.data_nsdq, i+input_t, output_t)
        # 관리가 용이하도록 dictionary 형태로 모으기
        output_dic = {'spy': output_spy, 'tlt': output_tlt, 'oil': output_oil, 'gold': output_gold, 'nsdq': output_nsdq}

        date = self.data_spy.index[i+output_t].strftime('%Y-%m-%d')

        return  date, input_dic, output_dic

vDataset = vStockDataset()
vDataset.data_pre_process()

vDataset.data_spy["Adj Close"][1:].plot()
vDataset.data_tlt["Adj Close"][1:].plot()
vDataset.data_gold["Adj Close"][1:].plot()
vDataset.data_oil["Adj Close"][1:].plot()
vDataset.data_nsdq["Adj Close"][1:].plot()
plt.show()
##
with torch.no_grad():
    # 예측
    pred = STOCKMODEL(snp_start_val, bond_start_val, gold_start_val)


##

