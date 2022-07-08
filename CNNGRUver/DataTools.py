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

from datetime import timedelta
def append_time_step(x, y, time_step, time_term):
    z = []
    for i in range(time_step, len(x) - time_term):
        inputs = x.iloc[i - time_step: i]             # time step에 맞게 input만들기
        y_index = y.index.get_loc(inputs.index[-1])   # 각 time step 마지막날 추출
        outputs = y.iloc[[y_index+time_term], :]           # 추출한 마지막날+time_term영업일 추가
        merged = pd.concat([inputs, outputs])
        z.append(merged)
    return z

def pct_change_except_bond(stk, period=1):
    df = stk.copy()
    #multi index column이라면 column index를 flatten
    df.columns = pd.Index([e[0] + e[1] for e in df.columns.tolist()])

    # 채권 빼고 변화율로..
    for i in df.columns:
        if not 'TNX' in i:
            df[i] = df[i].pct_change(period) * 100

    return df.iloc[period:]
