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
from sklearn.preprocessing import OneHotEncoder

def read_file():
    start_date = pd.to_datetime('2020-01-01')
    end_date = pd.to_datetime('2021-03-21')
    freq = 'D'
    index_name = '日期'

    data_fp = os.getcwd() + os.sep + 'dataset' + os.sep
    file_list = os.listdir(data_fp)
    
    date = pd.date_range(start=start_date, end=end_date, freq=freq)
    data_df = pd.DataFrame(index=date)
    data_df.index.name = index_name
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
            continue
            
            
        try:
            df.index = pd.to_datetime(df.index, format='%Y%m%d')
        except:
            try:
                df.index = pd.to_datetime(df.index)
            except:
                raise Exception
          
        df.columns = [f'{col} - {file}' for col in df.columns]
        data_df = data_df.merge(df, left_index=True, right_index=True, how='left')
    
    # # Add additional date columns and do one-hot-encoding

    # data_df['Weekday'] = data_df.index.dayofweek # Monday=0, Sunday=6
    # data_df['Month'] = data_df.index.month # January=1, December=12
    
    # temp = np.array(list(data_df.index.strftime('%A'))).reshape(-1, 1)
    # weekday_ohe = OneHotEncoder().fit(temp)
    # data_df[weekday_ohe.get_feature_names(['Weekday'])] \
    #     = weekday_ohe.transform(temp).toarray()

    # temp = np.array(list(data_df.index.month_name())).reshape(-1, 1)
    # month_ohe = OneHotEncoder().fit(temp)
    # data_df[month_ohe.get_feature_names(['Month'])] \
    #     = month_ohe.fit_transform(temp).toarray()

    # Transform into numercal data
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
   
# Read file
data_df = read_file()


root_fp = os.getcwd() + os.sep
data_df.to_csv(root_fp+'training_data.csv', encoding='big5')

