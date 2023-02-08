import pandas as pd
import FinanceDataReader as fdr
import matplotlib.pyplot as plt

hsi = fdr.DataReader('HSI')
kospi = fdr.DataReader('KS11')
krw_usd = fdr.DataReader('FRED:DEXKOUS')
t10y_2y = fdr.DataReader('FRED:T10Y2Y')
t10 = fdr.DataReader('FRED:DGS10')
#tk10 = fdr.DataReader('KR10YT=RR')

##
'''
import matplotlib.pyplot as plt

hsi.plot(y='Open', color='red', label='HSI', title='Stock Indices')
kospi.plot(y='Open', color='blue', label='KOSPI', title='Stock Indices')
plt.legend()
krw_usd.plot(y='DEXKOUS', color='green', label='KRW/USD', title='Exchange Rates')
t10y_2y.plot(y='T10Y2Y', color='purple', label='10-Year Treasury to 2-Year Treasury', title='Yield Curves')
t10.plot(y='DGS10', color='orange', label='10 year Treasury', title='10Y Yield')
plt.show()
'''


##
start_date = '1996-12-13'
def normalize(df):
    df = (df - df.min()) / (df.max() - df.min())
    return df

def stock_diff(df, period=1, day = '1996-12-13'):
    df = df.dropna()
    stock_diff = df.pct_change(period)
    # Normalize
    volume_norm = normalize(df['Volume'])
    stock_diff['Volume'] = volume_norm
    return stock_diff[stock_diff.index>=day]

hsi_1 = stock_diff(hsi, 1, start_date)
kospi_1 = stock_diff(kospi, 1, start_date)
hsi_5 = stock_diff(hsi, 5, start_date)
kospi_5 = stock_diff(kospi, 5, start_date)
t10_ = normalize(t10)[t10.index>=start_date]
krw_usd_ = (krw_usd/krw_usd.max())[krw_usd.index>=start_date]
t10y_2y_ = (t10y_2y/t10y_2y.max())[t10y_2y.index>=start_date]

ax = hsi_1.plot(y='Open', color='red', label='HSI', alpha=0.5)
hsi_1.plot(y='Volume', color='pink', label='HSI volume', alpha=0.5, ax=ax)
kospi_1.plot(y='Open', color='blue', label='KOSPI', alpha=0.5, ax=ax)
kospi_1.plot(y='Volume', color='skyblue', label='KOSPI volume', alpha=0.5, ax=ax)
plt.title('Stock Indices')
plt.legend()
plt.show()

ax1 = krw_usd_.plot(y=krw_usd_.columns[0], color='red', label='KRW/USD', alpha=0.5)
t10_.plot(y=t10_.columns[0], color='blue', label='T10', alpha=0.5, ax=ax1)
t10y_2y_.plot(y=t10y_2y_.columns[0], color='green', label='T10Y-2Y', alpha=0.5, ax=ax1)
plt.axhline(y=0, color='black', linestyle='-')
plt.title('Economic Indicators')
plt.legend()
plt.show()

##

ax1 = plt.hist([kospi_5.Open],range=[-0.1, 0.1], bins=1000,alpha=0.5)
ax2 = plt.hist([hsi_5.Open],range=[-0.1, 0.1], bins=1000,alpha=0.5)
plt.grid()
plt.show()
##
