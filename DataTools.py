import yfinance as yf

def data_import(start_date='2003-01-01', end_date='2030-12-31'):
    stock_spy = yf.download('^GSPC',start_date,end_date)
    stock_nsdq = yf.download('^IXIC',start_date,end_date)
    stock_tlt = yf.download('ZB=F',start_date,end_date)
    stock_gold = yf.download('GC=F',start_date,end_date)
    stock_oil = yf.download('CL=F',start_date,end_date)
    
    # 데이터 간격 맞추기: 빈날=전날값
    # self.data_gold = self.data_gold.reindex(self.data_spy.index, fill_value=0)
    stock_spy = stock_spy.reindex(stock_nsdq.index, method='ffill')
    stock_nsdq = stock_nsdq.reindex(stock_nsdq.index, method='ffill')
    stock_tlt = stock_tlt.reindex(stock_nsdq.index, method='ffill')
    stock_gold = stock_gold.reindex(stock_nsdq.index, method='ffill')
    stock_oil = stock_oil.reindex(stock_nsdq.index, method='ffill')

    return {'spy':stock_spy, 'nsdq':stock_nsdq, 'tlt':stock_tlt, 'gold':stock_gold, 'oil':stock_oil}

def split(data, start_date, end_date):

    output={}
    for i in data:
        output[i] = data[i].loc[start_date:end_date]

    return output

def to_percentage(data):
    for i in range(1, len(data)):
        # 볼륨값을 제외한 나머지는 맨 뒤에서부터 앞에값의 차이로 구한다
        temp = data.iloc[-i]  # 맨뒤
        temp2 = data.iloc[-(i + 1)]  # 그 전날
        newtemp = ((temp[0:6] / temp2[0:5]) - 1) * 100  # 당일/전날 하여 전날대비상승률 계산, 1을 빼서 정확한 %계산
        newtemp['Volume'] = temp['Volume']  # 볼륨값 유지
        data.iloc[-i] = newtemp
    # Volume 평준화 시행
    data['Volume'] = data['Volume'] / data['Volume'].max()

    return data.iloc[1:]

def to_percentage_period(data, period):

    if len(data)<period: print('WARNING: period is bigger than data length')

    for i in range(1, len(data)-period):
        temp = data.iloc[-i]  # 맨뒤
        temp2 = data.iloc[-(i + period)]
        newtemp = ((temp[0:6] / temp2[0:5]) - 1) * 100
        newtemp['Volume'] = temp['Volume']
        data.iloc[-i] = newtemp
    data['Volume'] = data['Volume'] / data['Volume'].max()
    return data.iloc[period+1:]

def data_pre_process_(data):
    print('Pre processing...')
    data['spy'] = to_percentage(data['spy'])
    data['tlt'] = to_percentage(data['tlt'])
    data['gold'] = to_percentage(data['gold'])
    data['nsdq'] = to_percentage(data['nsdq'])
    data['oil']['Volume'] = data['oil']['Volume'] / data['oil']['Volume'].max()
    print('Done')

def data_pre_process_period(data, period=5):
    print('Calculating Period diff...')
    ret = data.copy()
    ret['spy'] = to_percentage_period(ret['spy'],period)
    ret['tlt'] = to_percentage_period(ret['tlt'],period)
    ret['gold'] = to_percentage_period(ret['gold'],period)
    ret['nsdq'] = to_percentage_period(ret['nsdq'],period)
    ret['oil']['Volume'] = ret['oil']['Volume'] / ret['oil']['Volume'].max()
    print('Done')
    return ret



##

