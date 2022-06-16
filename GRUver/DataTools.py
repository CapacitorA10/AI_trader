import yfinance as yf
import pandas as pd
import numpy as np
def data_import(start_date='2003-01-01', end_date='2030-12-31', item=['^IXIC','^TNX','GC=F']):
    stock = yf.download(item, start_date, end_date, group_by = 'column')
    stock = stock.drop('Volume', axis=1)
    stock = stock.drop('High', axis=1)
    stock = stock.drop('Low', axis=1)
    stock = stock.drop('Adj Close', axis=1)
    # 데이터 간격 맞추기: 빈날=전날값
    #stock = stock.reindex(stock.index, method='ffill')
    stock = stock.fillna(method='ffill')
    return stock

def split(data, start_date, end_date):
    output = {}
    for i in data:
        output[i] = data[i].loc[start_date:end_date]

    return output

def append_time_step(data, time_step):
    output = []
    for i in range(time_step, len(data) - 1):
        output.append(data[i - time_step: i])
    return output

def pct_change_except_bond(stk, period=1):
    df = stk.copy()
    #multi index column이라면 column index를 flatten
    df.columns = pd.Index([e[0] + e[1] for e in df.columns.tolist()])

    # 채권 빼고 변화율로..
    for i in df.columns:
        if not 'TNX' in i:
            df[i] = df[i].pct_change(period) * 100

    return df.iloc[period:]

import torch
def pullout(data, idx, idx2):
    # pytorch가 이용 가능한 형태로 데이터 추출(index 및 기간 설정)
    # permute 사용해서 색인 축(open,close등)과 day축을 교환
    return (torch.from_numpy(data.iloc[idx:idx2].values).float()).permute(1, 0)
##
