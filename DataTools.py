import yfinance as yf

def data_import(start_date='2003-01-01', end_date='2030-12-31'):
    stock_spy = yf.download('SPY',start_date,end_date)
    stock_nsdq = yf.download('NQ=F',start_date,end_date)
    stock_tlt = yf.download('TLT',start_date,end_date)
    stock_gold = yf.download('GC=F',start_date,end_date)
    stock_oil = yf.download('CL=F',start_date,end_date)

    return {'spy':stock_spy, 'nsdq':stock_nsdq, 'tlt':stock_tlt, 'gold':stock_gold, 'oil':stock_oil}


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
