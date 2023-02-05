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
def normalize(df):
    df = (df - df.min()) / (df.max() - df.min())
    return df

def stock_diff(df):
    df = df.drop(['Volume'], axis=1).dropna()
    df = df.pct_change()
    df = normalize(df)
    df = df.apply(lambda x: 2 * x - 1)
    return df

##
import matplotlib.pyplot as plt

hsi_diff = stock_diff(hsi)
kospi_diff = stock_diff(kospi)
t10_ = normalize(t10)
krw_usd_ = krw_usd/krw_usd.max()
t10y_2y_ = t10y_2y/t10y_2y.max()

ax = hsi_diff.plot(y='Open', color='red', label='HSI', alpha=0.5)
kospi_diff.plot(y='Open', color='blue', label='KOSPI', alpha=0.5, ax=ax)
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


##

