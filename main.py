##
import ai_trader.DataTools as DTs
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

input_t = 6 # 입력데이터 period 100 - 70 - 30 - 15 - 6 등...
output_t = 1 # 출력데이터 길이
period = 5 # 입력-출력간 기간
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
        self.x_spy = stock_x['spy']
        self.x_nsdq = stock_x['nsdq']
        self.x_tlt = stock_x['tlt']
        self.x_oil = stock_x['oil']
        self.x_gold = stock_x['gold']

        self.y_spy = stock_y['spy']
        self.y_nsdq = stock_y['nsdq']
        self.y_tlt = stock_y['tlt']
        self.y_oil = stock_y['oil']
        self.y_gold = stock_y['gold']

    def __len__(self):
        return len(self.x_spy) - input_t - output_t - period + 1

    def __getitem__(self, i):

        # pytorch가 이용 가능한 형태로 데이터 추출(index 및 기간 설정)
        def pullout(data,idx,period):
            # permute 사용해서 색인 축(open,close등)과 day축을 교환
            return (torch.from_numpy(data.iloc[idx:idx+period].values).float()).permute(1,0)

        input_spy = pullout(self.x_spy, i, input_t)
        input_tlt = pullout(self.x_tlt, i, input_t)
        input_oil = pullout(self.x_oil, i, input_t)
        input_gold = pullout(self.x_gold, i, input_t)
        input_nsdq = pullout(self.x_nsdq, i, input_t)
        input_dic = {'spy':input_spy, 'tlt':input_tlt, 'oil':input_oil, 'gold':input_gold, 'nsdq':input_nsdq}

        output_spy = pullout(self.y_spy, i+input_t, output_t) # 출력값은 입력한 날 바로 다음날부터 가져옴!
        output_tlt = pullout(self.y_tlt, i+input_t, output_t)
        output_oil = pullout(self.y_oil, i+input_t, output_t)
        output_gold = pullout(self.y_gold, i+input_t, output_t)
        output_nsdq = pullout(self.y_nsdq, i+input_t, output_t)
        output_dic = {'spy': output_spy, 'tlt': output_tlt, 'oil': output_oil, 'gold': output_gold, 'nsdq': output_nsdq}

        date = self.y_spy.index[i+input_t+output_t-1].strftime('%Y-%m-%d')

        return  date, input_dic, output_dic

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
STOCKMODEL.train()
optimizer = torch.optim.Adam(STOCKMODEL.parameters(), lr=0.0001)
trainLoader = DataLoader(dataset=trainData, batch_size=1, shuffle=True, num_workers=0)
testLoader = DataLoader(dataset=testData, batch_size=1, shuffle=False, num_workers=0)

# tensorboard 그리기
_, sample, _ = next(iter(trainLoader))
writer.add_graph(STOCKMODEL,[sample['spy'].to(device)
                            ,sample['tlt'].to(device)
                            ,sample['gold'].to(device)
                            ,sample['oil'].to(device)
                            ,sample['nsdq'].to(device)])
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
        optimizer.zero_grad()
        pred = STOCKMODEL(inVal['spy'].to(device),
                          inVal['tlt'].to(device),
                          inVal['gold'].to(device),
                          inVal['oil'].to(device),
                          inVal['nsdq'].to(device)).squeeze()
        # open/high/low/close/adjusted/volume
        ans_spy = outVal['spy'][0][0].to(device)
        cost = F.l1_loss(pred, ans_spy.squeeze(), reduction="none").to(device)
        cost.backward(cost)
        optimizer.step()
        # LOSS Calc
        loss += cost.sum() / output_t
        writer.add_scalar("Loss/train", cost.sum() / output_t, tb_step)

        # verification
        vLoss = 0
        step2 = 0
        if step % 2500 == 1:
            with torch.no_grad():
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


##
