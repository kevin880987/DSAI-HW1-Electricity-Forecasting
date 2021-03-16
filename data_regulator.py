# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:55:25 2021

@author: kevin
"""

import numpy as np
import pandas as pd
# import json
import os

from datetime import timedelta


def read_file():
    start_date = pd.to_datetime('2020-01-01')
    end_date = pd.to_datetime('2021-03-15')
    freq = 'D'

    data_fp = os.getcwd() + os.sep + 'dataset' + os.sep
    file_list = os.listdir(data_fp)
    
    date = pd.date_range(start=start_date, end=end_date, freq=freq)
    data_df = pd.DataFrame(index=date)
    for file in file_list:
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(data_fp+file, encoding='utf-8')
            except:
                try:
                    df = pd.read_csv(data_fp+file, encoding='Big5')
                except:
                    raise Exception
        
        else:
            continue
            
        if file=='MODIFIED.csv':
            df.set_index(df.columns[0], inplace=True)
            
        # if file in ['台灣電力公司_過去電力供需資訊.csv', 
        #             '本年度每日尖峰備轉容量率.csv', 
        #             '近三年每日尖峰備轉容量率.csv']:
        #     df.set_index(df.columns[0], inplace=True)
            
        # elif file in ['經濟部能源局_電力供需表.csv']:
        #     df.iloc[:, 0] += 191100
        #     df.set_index(df.columns[0], inplace=True)
        #     df.index = pd.to_datetime(df.index, format='%Y%m') + timedelta(days=14)
        #     df = df.iloc[:, 1:]
    
        # elif file in ['經濟部能源局_未來電力供需預測.csv']:
        #     df.iloc[:, 0] += 1911
        #     df.set_index(df.columns[0], inplace=True)
        #     df.index = pd.to_datetime(df.index, format='%Y') + timedelta(days=183)
    
        # elif file in ['經濟部能源局_國內歷次調整之電價.csv']:
        #     continue
        #     # df.set_index(df.columns[0], inplace=True)
        #     # df.index = pd.to_datetime(df.index, format='%Y%m%d')
        #     # df = df[(df.index>=start_date) & (df.index<=end_date)]
    
        #     # temp_df = pd.DataFrame(index=date)
        #     # for col, temp in df.groupby('項目'):
        #     #     if col not in temp_df:
        #     #         temp_df[col] = np.nan
                    
        #     #     temp_df.loc[temp.index, col] = temp.values
            
        # elif file in ['科技部科學園區用電負載量.csv']:
        #     df = df.transpose()
        #     df = df.iloc[1: , 2: 118]
        #     df.set_index(df.columns[0], inplace=True)
        #     df.set_axis(pd.MultiIndex.from_frame(df.iloc[0: 2, : ].transpose()), axis=1, inplace=True)
        #     df = df[2: ]
    
        #     temp_df = pd.DataFrame(index=date)
        #     for (c, y), ser in df.items():
        #         if c==c:
        #             col = c
        #         elif col!=col:
        #             raise ValueError
                    
        #         if col not in temp_df:
        #             temp_df[col] = np.nan
                
        #         sd = f'{int(y[: -1])+1911}-01-01'
        #         ed = f'{int(y[: -1])+1911}-12-31'
        #         if sd not in date or ed not in date:
        #             continue
        #         d = pd.date_range(start=sd, end=ed, freq='MS') + timedelta(days=14)
    
        #         temp_df.loc[d, col] = ser.values
            
        #     df = temp_df
            
            
        # Weather data
        
        
        else:
            print(file, 'skipped')
            
            
        try:
            df.index = pd.to_datetime(df.index, format='%Y%m%d')
        except:
            try:
                df.index = pd.to_datetime(df.index)
            except:
                raise Exception
          
        df.columns = [f'{col} - {file}' for col in df.columns]
        data_df = data_df.merge(df, left_index=True, right_index=True, how='left')
        
    # Transform
    for col, ser in data_df.items():
        if np.any(ser==ser):
            if not np.issubdtype(ser, np.number):
                data_df[col] = ser.astype(float).values
                # print(col)
                # # Replace string with nan
                # df[col].replace(str_to_nan_list, np.nan, inplace=True)
    
        else:
            data_df.drop(col, axis=1, inplace=True)
    
    return data_df

# df=data_df
def amortize(df):
    """
    Input a day-wise sparse dataframe.
    Return an amortized dataframe.

    Parameters
    ----------
    df : dataframe
        A sparse dataframe with date as its index.
        
        e.g.
          DATE  Brent Oil Futures Historical Data - Price
    2010-01-01                                        NaN
    2010-01-02                                        NaN
    2010-01-03                                        NaN
    2010-01-04                                      80.12
    2010-01-05                                      80.59

    Par : dictionary
        Costomized parameters imported from 'parameters.py'.

    Raises
    ------
    ValueError
        Raised when the amortization contains NaN.

    Returns
    -------
    df : dataframe
        A dataframe with no NaN and date as its index.
        
        e.g.
          DATE  Brent Oil Futures Historical Data - Price
    2010-01-01                                      80.12
    2010-01-02                                      80.12
    2010-01-03                                      80.12
    2010-01-04                                      80.12
    2010-01-05                                      80.59

    """
    
    display, verbose = True, True
    if display:
        feature_ctr, unab_amort_list = 0, []

    df = df.copy()
    
    for col in df.columns:
        # if verbose:
        #     print(col)

        index = np.where(df[col].notnull())[0]
        if index.size >= 2:
            amortization = [df[col].iloc[index[0]]] * (index[0] - 0)
            for i in range(len(index)-1):
                amortization.extend(
                    np.linspace(float(df[col].iloc[index[i]]), 
                                float(df[col].iloc[index[i+1]]), 
                                index[i+1]-index[i], endpoint=False)
                    )    

                if np.any(pd.isnull(amortization)):
                    print(i)
                    raise ValueError(f'{col} contains NaN')

            amortization.extend(
                [df[col].iloc[index[i+1]]] * (len(df[col]) - 1 - index[i+1] + 1)
                )
                    
            df[col] = amortization
            
            # Make sure all values are converted into number
            df[col] = df[col].astype(float)
            
            if np.any(pd.isnull(df[col])):
                print('null', col)
                raise ValueError
            
            if display:
                feature_ctr += 1
            
        elif index.size < 2:
            if display:
                unab_amort_list.append(col)
            if verbose:
                print(f'Unable to amortize {col}')
                
            df.drop(columns=col, inplace=True)
            
    return df

        

def split_train_test(data_df, target=[]):
    train_step = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    test_step = np.array([1, 2, 3, 4, 5, 6, 7])
    index = data_df.index
    
    columns = data_df.columns    
    training_x_df = pd.DataFrame(index=index)
    for step in train_step:
        training_x_df[[f'{c} - (-{step})' for c in columns]] = data_df[columns].shift(step)
    training_x_df.dropna(axis=0, inplace=True)
    
    training_y_df = pd.DataFrame(index=index)
    for step in test_step:
        training_y_df[[f'{t} - (+{step})' for t in target]] = data_df[target].shift(-step)
    training_y_df.dropna(axis=0, inplace=True)
    
    index = sorted(list(set(training_x_df.index) & set(training_y_df.index)))
    training_x_df = training_x_df.loc[index]
    training_y_df = training_y_df.loc[index]
    
    return training_x_df, training_y_df

   
# Read file
data_df = read_file()

# Amortize
data_df = amortize(data_df)

# Split
training_x_df, training_y_df = split_train_test(data_df, target=['備轉容量(MW) - MODIFIED.csv'])

root_fp = os.getcwd() + os.sep
training_x_df.to_csv(root_fp+'training_data.csv', encoding='big5')
training_y_df.to_csv(root_fp+'training_data_y.csv', encoding='big5')

# Y for Y
# target_df = data_df['備轉容量(MW) - 台灣電力公司_過去電力供需資訊.csv']
# target_df.plot()

#     elif file.endswith('.json'):break
#         df = pd.read_json(data_fp+file, encoding='UTF-8')

#         with open(data_fp+file, encoding='utf-8') as f:
#             data = f.read()
        
#         d = eval(data)
#         dd = pd.json_normalize(data['cwbdata'])

# file='觀測網旬資料-農業氣象觀測網旬資料.json'


# import urllib.request, json, ssl
# url = 'http://data.taipower.com.tw/opendata/apply/file/d006009/001.json'

# url = 'https://opendata.cwb.gov.tw/dataset/climate/C-A0008-001'
# context = ssl._create_unverified_context()
# with urllib.request.urlopen(url, context=context) as jsondata:
#     data = json.loads(jsondata.read().decode())
# # print(data)
# d=data['cwbdata']['resources']['resource']['data']['agrObs']['location']
# d=pd.json_normalize(d)
