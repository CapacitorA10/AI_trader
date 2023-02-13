import numpy as np
import pandas as pd

def normalize(df):
    df = (df - df.min()) / (df.max() - df.min())
    return df


def stock_diff(df, period, day='1996-12-13'):
    df = df.dropna()
    stock_diff = df.pct_change(period)
    # Normalize
    volume_norm = normalize(df['Volume'])
    stock_diff['Volume'] = volume_norm
    stock_diff = stock_diff[stock_diff.index >= day]
    return stock_diff.iloc[period:]

def merge_dataframes(input_df_list, output_df_list, flag):
    merged_df = pd.concat(input_df_list + output_df_list, axis=1, sort=False)
    if flag == "ffill":
        merged_df.fillna(method="ffill", inplace=True)
    elif flag == "bfill":
        merged_df.fillna(method="bfill", inplace=True)
    elif flag == "drop":
        merged_df.dropna(inplace=True)

    output_df_cols = sum([df.shape[1] for df in output_df_list])
    #print("Length of dataframes' columns inside output_df_list:", output_df_cols)
    return merged_df, output_df_cols

def append_time_step(df, training_span, cum_volatility, split):
    z = []
    for i in range(training_span, len(df) - cum_volatility):
        x = df.iloc[i - training_span: i, :split]  # time step에 맞게 input만들기
        y = df.iloc[i + cum_volatility, split:]  # 추출한 마지막날+time_term영업일 추가
        merged = pd.concat([x, y.to_frame().T], axis=0)
        z.append(merged)
    return z