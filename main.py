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
        return len(self.data_spy)-input_t-output_t

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
trainLoader = DataLoader(dataset=trainData, batch_size=1, shuffle=False)
testLoader = DataLoader(dataset=testData, batch_size=1, shuffle=False)
max_epoch = 15
tb_step = 0
tb_step2 = 0
for epoch in range(max_epoch):
    loss = 0
    step = 0
    for date, inVal, outVal in trainLoader:
        step += 1
        tb_step += 1
        '''
        if date[0] == '2019-07-31': #2003 - 2019년 여름까지만 학습
            print('load done')
            break
        '''
        # LEARNING START
        optimizer.zero_grad()
        pred = STOCKMODEL(inVal['spy'].to(device),
                          inVal['tlt'].to(device),
                          inVal['gold'].to(device),
                          inVal['oil'].to(device),
                          inVal['nsdq'].to(device)).squeeze()
        # open/high/low/close/adjusted/volume
        ans_spy = outVal['spy'][0][0].to(device)
        cost = F.l1_loss(pred, ans_spy, reduction="none").to(device)
        cost.backward(cost)
        optimizer.step()
        # LOSS Calc
        loss += abs(cost).sum()
        writer.add_scalar("Loss/train", abs(cost).sum(), tb_step)

        # verification

        if step % 1000 == 1:
            with torch.no_grad():
                for vDate, vInVal, vOutVal in testLoader:
                    '''
                    if vDate[0] == '2021-11-05':  # 2003 - 2019년 여름까지만 학습
                        print('verification done')
                        break
                        '''
                    vPred = STOCKMODEL(vInVal['spy'].to(device),
                                       vInVal['tlt'].to(device),
                                       vInVal['gold'].to(device),
                                       vInVal['oil'].to(device),
                                       vInVal['nsdq'].to(device)).squeeze()
                    vAns_spy = vOutVal['spy'][0][0].to(device)
                    vCost = F.l1_loss(vPred, vAns_spy, reduction="none").to(device)
                    writer.add_scalar("Loss/Test", abs(vCost).sum(), tb_step2)
                    tb_step2 += 1
                print(f"verification done, Last date = {vDate}")

    loss /= step
    print(f"Learning done, Last date = {date}")
    print(f"epoch{epoch} mean loss: {loss}")

#######################################################################################################################
################################################## 여기서부터 검증 ######################################################

