### Package Loading

# Base
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

# PCA
from sklearn.decomposition import PCA

def PCA_result(train, test,n_components,origin) :
    """
    PCA를 위한 함수
    ---input---
    train : train 데이터
    test : test 데이터
    n_components : PCA의 n_components
    origin : 정제하지 않은 train 데이터
    ---output---
    data : 라벨과 시간정보가 포함된 PCA된 train 데이터
    value : 라벨과 시간정보가 포함되지 않은 PCA된 train 데이터
    test : 라벨과 시간정보가 포함되지 않은 PCA된 test 데이터
    """
    
    # PCA Fitting
    model = PCA()
    model_fit = model.fit(train)

    # Elbow Plot
    print("Elbow Plot")
    evr = round(pd.DataFrame(model_fit.explained_variance_ratio_),5) * 100
    evr = evr.reset_index()
    plt.figure(figsize = (8,5))
    plt.scatter(evr["index"],(evr[0]/np.sum(evr[0]))*100)
    plt.plot((evr[0]/np.sum(evr[0]))*100)
    plt.hlines(5,xmin = 0,xmax = 52,color="red")
    plt.show()

    # transform
    value = model_fit.fit_transform(train)
    test = model_fit.fit_transform(test)

    # extract n components
    value = pd.DataFrame(value).iloc[:,0:n_components]
    test = pd.DataFrame(test).iloc[:,0:n_components]

    # Make DataSet For PCA Plot
    data = pd.concat([origin["timestamp"],value,origin["machine_status"]], axis = 1)
    data.rename(columns = {0:"PC1",1:"PC2",2:"PC3"},inplace = True)

    # PCA 3D Plot
    print("PCA 3D Plot")
    fig = plt.figure(figsize = (20, 10))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(data[data["machine_status"] == "NORMAL"]['PC1'], data[data["machine_status"] == "NORMAL"]['PC2'], data[data["machine_status"] == "NORMAL"]['PC3'],c = "blue")
    ax.scatter3D(data[data["machine_status"] ==  'RECOVERING']['PC1'], data[data["machine_status"] ==  'RECOVERING']['PC2'], data[data["machine_status"] ==  'RECOVERING']['PC3'],c = "yellow")
    plt.title("PCA")
    plt.show()

    # PCA 2D Plot(PC1, PC2)
    print("PCA 2D Plot(PC1, PC2)")
    plt.figure(figsize = (10,6))
    plt.scatter(data[data["machine_status"] == "NORMAL"]['PC1'], data[data["machine_status"] == "NORMAL"]['PC2'],color = "blue")
    plt.scatter(data[data["machine_status"] ==  'RECOVERING']['PC1'], data[data["machine_status"] ==  'RECOVERING']['PC2'],color = "yellow")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

    # PCA 2D Plot(PC2, PC3)
    print("PCA 2D Plot(PC2, PC3)")
    plt.figure(figsize = (10,6))
    plt.scatter(data[data["machine_status"] == "NORMAL"]['PC2'], data[data["machine_status"] == "NORMAL"]['PC3'],color = "blue")
    plt.scatter(data[data["machine_status"] ==  'RECOVERING']['PC2'], data[data["machine_status"] ==  'RECOVERING']['PC3'],color = "yellow")
    plt.xlabel("PC2")
    plt.ylabel("PC3")
    plt.show()

    # PCA 2D Plot(PC1, PC3)
    print("PCA 2D Plot(PC1, PC3)")
    plt.figure(figsize = (10,6))
    plt.scatter(data[data["machine_status"] == "NORMAL"]['PC1'], data[data["machine_status"] == "NORMAL"]['PC3'],color = "blue")
    plt.scatter(data[data["machine_status"] ==  'RECOVERING']['PC1'], data[data["machine_status"] ==  'RECOVERING']['PC3'],color = "yellow")
    plt.xlabel("PC1")
    plt.ylabel("PC3")
    plt.show()

    return model_fit, data, value, test

def biplot(pca_model, data,labels = None) :
    """
    biplot을 그리기 위한 함수
    ---input---
    pca_model : pca를 fitting 시키는 함수
    data : pca를 한 train 데이터에 target과 timestamp가 존재하는 데이터
    ---output---
    plot : biplot 그래프
    """
    y = np.array(data[["machine_status"]].replace({"NORMAL":0,"RECOVERING":1}))
    X_new = data[["PC1","PC2","PC3"]]
    X_new = np.array(X_new)
    K = np.transpose(pca_model.components_[0:2,:])

    xs = X_new[:,0]
    ys = X_new[:,1]
    n = K.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    
    plt.figure(figsize = (40,40))
    plt.scatter(xs * scalex,ys * scaley, c = y)
    plt.xlim(-0.75,0.75)
    plt.ylim(-0.5,1)
    plt.grid()
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))

    for i in range(n):
        plt.arrow(0, 0, K[i,0], K[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(K[i,0]* 1.15, K[i,1] * 1.15, "Sensor"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(K[i,0]* 1.15, K[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    
    plt.show()



