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
#input_t = int(input("input length? "))# 입력데이터 period 100 - 70 - 30 - 15 - 6 등...
#period = int(input("period term? "))#입력-출력간 기간
output_t = 1 # 출력데이터 길이(채권, 나스닥, 금)
bSize = 32   # 배치 사이즈
torch.cuda.is_available()
## Data import

stocks = DTs.data_import('2000-09-01', '2025-01-01') #item변수 전달 안하면, 기본 3개 나스닥 채권 금만 return


stocks_1day_change = DTs.pct_change_except_bond(stocks)
stocks_15day_change =  DTs.pct_change_except_bond(stocks, 15)

##
#periodDiff_stocks = DTs.data_pre_process_period(stocks, period=period)
DTs.data_pre_process_(stocks)
stock_train_x = DTs.split(stocks, '2001-01-01', '2019-06-30')
stock_train_y = DTs.split(periodDiff_stocks, '2001-01-01', '2019-06-30')
stock_test_x = DTs.split(stocks, '2019-07-01', '2025-01-01')
stock_test_y = DTs.split(periodDiff_stocks, '2019-07-01', '2025-01-01')
##

