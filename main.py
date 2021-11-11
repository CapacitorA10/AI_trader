##
import DataTools as DTs
import torch
import torch.nn.init
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
device = 'cuda'
torch.cuda.is_available()
## data import
stocks = DTs.data_import('2003-01-01', '2021-11-09')
DTs.data_pre_process_(stocks)
stock_train = DTs.split(stocks, '2003-01-01', '2020-01-01')
stock_test = DTs.split(stocks, '2020-01-01', '2025-01-01')
## dataset 설정
input_t = 30 # 입력데이터 period
output_t = 10 # 출력데이터 period

class stockdataset(Dataset):
    def __init__(self, stock):
        self.data_spy = stock['spy']
        self.data_nsdq = stock['nsdq']
        self.data_tlt = stock['tlt']
        self.data_oil = stock['oil']
        self.data_gold = stock['gold']

    def __len__(self):
        return len(self.data_spy)

    def __getitem__(self, i):

        # pytorch가 이용 가능한 형태로 데이터 추출(index 및 기간 설정)
        def pullout(data,idx,period):
            # permute 사용해서 색인 축(open,close등)과 day축을 교환
            return (torch.from_numpy(data.iloc[idx:idx+period].values.astype(np.float64)).float()).permute(1,0)

        input_spy = pullout(self.data_spy, i, input_t)
        input_tlt = pullout(self.data_tlt, i, input_t)
        input_oil = pullout(self.data_oil, i, input_t)
        input_gold = pullout(self.data_gold, i, input_t)
        input_nsdq = pullout(self.data_nsdq, i, input_t)
        input_dic = {'spy':input_spy, 'tlt':input_tlt, 'oil':input_oil, 'gold':input_gold, 'nsdq':input_nsdq}

        output_spy = pullout(self.data_spy, i+input_t, output_t) # 출력값은 입력한 날 바로 다음날부터 가져옴!
        output_tlt = pullout(self.data_tlt, i+input_t, output_t)
        output_oil = pullout(self.data_oil, i+input_t, output_t)
        output_gold = pullout(self.data_gold, i+input_t, output_t)
        output_nsdq = pullout(self.data_nsdq, i+input_t, output_t)
        output_dic = {'spy': output_spy, 'tlt': output_tlt, 'oil': output_oil, 'gold': output_gold, 'nsdq': output_nsdq}

        date = self.data_spy.index[i+input_t+output_t-1].strftime('%Y-%m-%d')

        return  date, input_dic, output_dic

## data 전처리 및 show
trainData = stockdataset(stock_train)
testData = stockdataset(stock_test)
'''
trainData.data_spy.plot()
testData.data_spy.plot()
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

## train
STOCKMODEL = comb_model1().to(device)
STOCKMODEL.train()
optimizer = torch.optim.Adam(STOCKMODEL.parameters(), lr=0.0001)
loader = DataLoader(dataset=trainData, batch_size=1, shuffle=False)

max_epoch = 7
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
vstock_spy, vstock_nsdq, vstock_tlt, vstock_gold, vstock_oil = DTs.data_import(start_date, end_date)

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

        date = self.data_spy.index[i+input_t+output_t-1].strftime('%Y-%m-%d')

        return  date, input_dic, output_dic

vDataset = vStockDataset()
vDataset.data_pre_process()

vDataset.data_spy["Adj Close"][1:].plot()
vDataset.data_tlt["Adj Close"][1:].plot()
vDataset.data_gold["Adj Close"][1:].plot()
vDataset.data_oil["Adj Close"][1:].plot()
vDataset.data_nsdq["Adj Close"][1:].plot()
plt.show()
## 검증용

vLoader = DataLoader(dataset=vDataset, batch_size=1, shuffle=False)
STOCKMODEL.eval()

with torch.no_grad():
    max_epoch = 7
    k = 0
    for epoch in range(max_epoch):
        loss = 0
        step = 0
        for vdate, vInVal, vOutVal in vLoader:
            step += 1
            k += 1
            if vdate[0] == '2021-11-05':
                print('load done')
                break
            # input = open/high/low/close/adjusted/volume
            # batch, feature, time step 순 정리
            vInp_spy = vInVal['spy'].to(device)
            vInp_tlt = vInVal['tlt'].to(device)
            vInp_gold = vInVal['gold'].to(device)
            vInp_oil = vInVal['oil'].to(device)
            vInp_nsdq = vInVal['nsdq'].to(device)

            # open 만 예상
            ans_spy = vOutVal['spy'][0][0].to(device)

            # LEARNING START
            vPred = STOCKMODEL(vInp_spy, vInp_tlt, vInp_gold, vInp_oil, vInp_nsdq).squeeze()
            vCost = F.l1_loss(vPred, ans_spy, reduction="none").to(device)

            # LOSS Calc
            loss += abs(cost).sum()
            writer.add_scalar("Verify_Loss/train", abs(cost).sum(), k)

        loss /= step
        print(loss)

##

