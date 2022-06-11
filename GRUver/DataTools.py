import yfinance as yf
import pandas as pd

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


def pct_change_except_bond(stk, period=1):
    df = stk.copy()
    #multi index column이라면 flatten
    df.columns = pd.Index([e[0] + e[1] for e in df.columns.tolist()])

    # 채권 빼고 변화율로..
    for i in df.columns:
        if not 'TNX' in i:
            df[i] = df[i].pct_change(period) * 100

    return df

def data_pre_process_(data):
    print('Pre processing...')
    data['spy'] = to_percentage(data['spy'])
    data['tlt'] = to_percentage(data['tlt'])
    data['gold'] = to_percentage(data['gold'])
    print('Done')


import copy
def data_pre_process_period(data, period=5):
    print('Calculating Period diff...')
    ret = copy.deepcopy(data)
    ret['spy'] = to_percentage_period(ret['spy'], period)
    ret['tlt'] = to_percentage_period(ret['tlt'], period)
    ret['gold'] = to_percentage_period(ret['gold'], period)
    print('Done')
    return ret

import torch
def pullout(data, idx, idx2):
    # pytorch가 이용 가능한 형태로 데이터 추출(index 및 기간 설정)
    # permute 사용해서 색인 축(open,close등)과 day축을 교환
    return (torch.from_numpy(data.iloc[idx:idx2].values).float()).permute(1, 0)
##
