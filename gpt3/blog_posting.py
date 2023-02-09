import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

# tk10 = fdr.DataReader('KR10YT=RR')
hsi = fdr.DataReader('HSI')
kospi = fdr.DataReader('KS11')
krw_usd = fdr.DataReader('FRED:DEXKOUS')
t10y_2y = fdr.DataReader('FRED:T10Y2Y')
t10 = fdr.DataReader('FRED:DGS10')

training_span = 3
cum_volatility = 5
## plot
''' # data 받은 후 plot해보기
hsi.plot(y='Open', color='red', label='HSI', title='Stock Indices')
kospi.plot(y='Open', color='blue', label='KOSPI', title='Stock Indices')
plt.legend()
krw_usd.plot(y='DEXKOUS', color='green', label='KRW/USD', title='Exchange Rates')
t10y_2y.plot(y='T10Y2Y', color='purple', label='10-Year Treasury to 2-Year Treasury', title='Yield Curves')
t10.plot(y='DGS10', color='orange', label='10 year Treasury', title='10Y Yield')
plt.show()
'''

## data 정규화
start_date = '1996-12-13'


def normalize(df):
    df = (df - df.min()) / (df.max() - df.min())
    return df


def stock_diff(df, period=cum_volatility, day='1996-12-13'):
    df = df.dropna()
    stock_diff = df.pct_change(period)
    # Normalize
    volume_norm = normalize(df['Volume'])
    stock_diff['Volume'] = volume_norm
    stock_diff = stock_diff[stock_diff.index >= day]
    return stock_diff.iloc[period:]


hsi_1 = stock_diff(hsi, 1, start_date)
kospi_1 = stock_diff(kospi, 1, start_date)
hsi_5 = stock_diff(hsi, 5, start_date)
kospi_5 = stock_diff(kospi, 5, start_date)
t10_ = normalize(t10)[t10.index >= start_date]
krw_usd_ = (krw_usd / krw_usd.max())[krw_usd.index >= start_date]
t10y_2y_ = (t10y_2y / t10y_2y.max())[t10y_2y.index >= start_date]
## plot
''' 데이터 후가공 후 plot해보기
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
'''


##
def merge_dataframes(input_df_list, output_df_list, flag):
    merged_df = pd.concat(input_df_list + output_df_list, axis=1, sort=False)
    if flag == "ffill":
        merged_df.fillna(method="ffill", inplace=True)
    elif flag == "bfill":
        merged_df.fillna(method="bfill", inplace=True)
    elif flag == "drop":
        merged_df.dropna(inplace=True)

    output_df_cols = sum([df.shape[1] for df in output_df_list])
    print("Length of dataframes' columns inside output_df_list:", output_df_cols)
    return merged_df, output_df_cols


##
stock_all, split = merge_dataframes([kospi_1], [kospi_5], "drop")


##
def append_time_step(df, training_span, cum_volatility, split):
    z = []
    for i in range(training_span, len(df) - cum_volatility):
        x = df.iloc[i - training_span: i, :split]  # time step에 맞게 input만들기
        y = df.iloc[i + cum_volatility, split:]  # 추출한 마지막날+time_term영업일 추가
        merged = pd.concat([x, y.to_frame().T], axis=0)
        z.append(merged)
    return z


##
stock_all_ = append_time_step(stock_all, training_span, cum_volatility, split)
stock_all_tensor = torch.Tensor(np.array(stock_all_))
train = stock_all_tensor[:-500, :, :]
test = stock_all_tensor[-500:, :, :]


##
# Create a custom dataset class that wraps the tensor
class CustomDataset(data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index, :, :]

    def __len__(self):
        return self.tensor.size(0)


# Create an instance of the custom dataset
dataset = CustomDataset(train)
# Create a DataLoader with shuffle=True, but only shuffle the first dimension
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

##

d
