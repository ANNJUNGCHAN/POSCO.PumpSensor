### Package Loading

# Base
from ast import Pass
import numpy as np
import pandas as pd

# Visualization
from matplotlib import pyplot as plt

# Data Preprocessing
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# Random and path
import os
import random

# 에러 메세지 필터링을 위해서 warnings 를 사용
import warnings
warnings.filterwarnings('ignore')


# Data Ploting

def plot_sensor(sensor, path) :
    """
    sensor별 plot을 그려주는 함수
    ---input---
    sensor : 센서 이름 ex) sensor_00
    path : 데이터 경로
    ---output---
    sensor에 대한 플랏
    """

    data = pd.read_csv(path)
    if "Unnamed: 0" in data.columns :
        data.drop(columns = ["Unnamed: 0"],inplace = True)
    else :
        pass
    
    data[sensor].plot(figsize = (10,2), title = sensor)

# Train Test Split & Data preprocessing

def DataPreprocessing(path,scaler = True) :
    """
    데이터 전처리 및 정제, Train Test 나누기
    1. 데이터를 불러온다.
    2. 데이터를 Error와 Normal로 이진화한다.
    3. train 60%, test 40%로 데이터를 나눈다.
    4. 시간열을 정제한다.
    5. 시간에 맞춰서 결측값을 보간한다.
    6. MinMaxScaler를 사용하여 데이터를 정규화한다.
    ---input---
    path : 데이터를 저장한 경로
    ---output---
    data : train 데이터
    test : test 데이터
    answer : test 데이터의 label
    origin : 정제하지 않은 train 데이터
    data_with_time : 정제된 train data에 timeseries가 붙어있음.
    """

    ### 데이터 불러오기
    data = pd.read_csv(path)
    if "Unnamed: 0" in data.columns :
        data.drop(columns = ["Unnamed: 0"],inplace = True)
    else :
        pass
    
    ### 데이터 이진화
    for i in range(len(data)) :
        if data["machine_status"][i] == "BROKEN":
            data["machine_status"][i] = "RECOVERING"
    
    ### Train Test Split and answer
    data = data.iloc[:int(len(data)*0.6)]
    test = data.iloc[int(len(data)*0.6):]
    answer = test["machine_status"]

    ### Handling TimeSeries Data
    data["timestamp"] = data["timestamp"].astype("datetime64")
    test["timestamp"] = test["timestamp"].astype("datetime64")

    ### Delete Useless column
    data.drop(columns = ["sensor_15"],inplace = True)
    test.drop(columns = ["sensor_15"],inplace = True)

    ### Remain Origin Train Data
    origin = data.copy()

    ### Interpolate missing data by Timeseries Interpolate
    data.index = data["timestamp"]
    test.index = test["timestamp"]
    data = data.drop(columns = "timestamp")
    test = test.drop(columns = "timestamp")
    data=data.interpolate(method = "time")
    test=test.interpolate(method = "time")
    data = data.reset_index()
    test = test.reset_index()
    data_with_time = data
    value = data.iloc[:,1:52]
    test = test.iloc[:,1:52]

    if scaler == True :    
        ## MinMax Scaler
        model = MinMaxScaler()
        model_fit = model.fit(value)
        value = model_fit.transform(value)
        test = model_fit.transform(test)
    else :
        pass
    
    ## Remain Normal Train Data
    data = pd.DataFrame(value)
    data.columns = origin.columns[1:52]
    data = pd.concat([data,origin["machine_status"]],axis = 1)
    data = data[data["machine_status"] == "NORMAL"].drop(columns = ["machine_status"])

    return data, test, answer, origin, data_with_time

# Interpolate Before After Plot
def Interpolate_plot(before,after,sensor) :
    """
    결측치 대체 후 데이터의 변화 비교
    ---input---
    before : 결측치 대체 전 데이터
    after : 결측치 대체 후 데이터
    sensor : 어떤 센서의 변화를 볼 것인가?
    * before와 after의 index는 같아야 하며, timestamp열이 존재해야한다.
    ---output---
    plot : 센서의 변화를 시각화한 plot
    """
    origin=before[["timestamp",sensor]]
    origin_right = origin[origin[sensor].isna() == True].index
    
    plt.figure(figsize = (20,6))
    plt.plot(origin["timestamp"],origin[sensor])
    plt.show()
    
    new=after[["timestamp",sensor]]
    plt.figure(figsize = (20,6))
    plt.plot(origin["timestamp"],origin[sensor])
    plt.scatter(new["timestamp"].iloc[origin_right],new[sensor].iloc[origin_right],color = "red")
    plt.show()

