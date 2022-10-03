### Loading Packages

# 에러 메세지 필터링을 위해서 warnings 를 사용
import warnings
warnings.filterwarnings('ignore')

# 오토인코더를 사용하기 위해서 torch를 사용
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn

# Base
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 랜덤시드 고정
import random
import os

# Scoring
from sklearn import metrics

### 랜덤시드 고정 함수
def set_seeds(seed):
    """
    랜덤시드를 고정하는 함수
    ---input---
    seed : 고정할 시드 넘버
    ---output---
    시드 고정
    """
    random_seed = seed
    SEED = seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

### AutoEncoder
def AutoEncoder(input_size, hidden_size, output_size):
    """
    AutoEncoder 정의 함수
    - 기본적인 오토인코더 모델을 정의
    - 실행한 후, 모델을 변수에 저장할 것
    ---input---
    input_size : input으로 들어가는 features 갯수 (해당 실험에서는 51 사용)
    hidden_size : Encoder/Decoder에서 hidden으로 얼마나 노드를 축소/확대할 것인가. (해당 실험에서는 30 사용)
    output_size : hidden에서 Encoder/Decoder의 Output으로 얼마나 노드를 축소/확대 할 것인가. (해당 실험에서는 20 사용)
    ---output---
    model : 오토인코더 모델
    """

    ### 모델 개형 정의
    class Model(nn.Module) :
        def __init__(self, input_size, hidden_size, output_size) :
            super(Model, self).__init__()

            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size

            # 인코더
            self.Encoder = nn.Sequential(
            
                nn.Linear(input_size,hidden_size[0]),
                nn.RReLU(),
                nn.Linear(hidden_size[0],output_size),
                nn.RReLU()
            )

            # 디코더
            self.Decoder = nn.Sequential(

                nn.Linear(output_size,hidden_size[0]),
                nn.RReLU(),
                nn.Linear(hidden_size[0],input_size)
            )

        # 전방 전달
        def forward(self, x) :
            x = self.Encoder(x)
            x = self.Decoder(x)

            return x

    # input_size, hidden_size, output_size 결정
    input_size = input_size
    hidden_size = [hidden_size]
    output_size = output_size

    # Model Define
    model = Model(input_size, hidden_size, output_size)

    return model

### Training Function
def training(model, data,loss_functions, learning_rate,batch_sizes,epochs):
    """
    Training 함수 정의
    - 옵티마이저, learning rate 정하기
    - 데이터로더 함수 정의
    - 에포크 정의
    ---input---
    model : 정의된 모델 넣기
    data : train data 넣기
    loss_fuction : 손실함수 정의 후 넣기 (실험시에는 nn.MSELoss() 사용)
    learning_rate : 학습률 정의 (실험시에는 0.01로 정의)
    batch_sizes : 배치 사이즈 정의 (실험시에는 64로 줌)
    epochs : 몇 에포크를 돌 것인지 정의(실험시에는 30번으로 줌)
    ---output---
    model : 학습된 모델
    """
    # Tensor 형태로 데이터 형 변환
    data = torch.Tensor(np.array(data))


    # loss function 정의
    loss_function = loss_functions
    
    # 옵티마이져는 Adam을 사용, learning rate는 0.01로 설정

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    

    # 데이터를 불러올 때, 배치사이즈는 64로 주고, shuffle = True로 설정하여 데이터를 뒤섞음
    dataloader = DataLoader(data, batch_size =batch_sizes, shuffle =True)
    
    for epoch in range(1, epochs+1): # 에포크 주기
        
        update_loss = 0.0
        
        for x in dataloader :
            optimizer.zero_grad()
            output = model(x)
            
            loss=loss_function(x, output) #dataloader로 불러온 데이터 값과 실제 데이터 간의 MSE 산출
            loss.backward()
            
            optimizer.step()
            update_loss += loss.item()

        print('epoch:', f'{epoch}', '  loss:', f'{update_loss}')
    
    return model

### Define Threshold
def Threshold(train,model,loss_function,threshold) :
    """
    오류로 판단할 임계값을 정의하는 함수
    - Threshold 이상이면 오류, 이하면 정상으로 판단하기 위해 Threshold럴 정하는 함수
    - 시각적으로 판단하면서, Threshold를 판단해보자.
    ---input---
    train : train data
    model : 학습된 모델
    loss_function : 에러 점수를 계산할 함수
    threshold : threshold로 정의할 숫자
    ---output---
    threshold plot : threshold를 정의하기 쉬운 plot
    """

    ### Error Point : 에러점수를 계산
    
    train_error = []
    for data in train :
        output = model(data)
        loss = loss_function(output, data)
        train_error.append(loss.item())

    ### Plotting
    plt.figure(figsize = (10,2))
    plt.scatter(np.arange(0,len(train_error)),np.array(train_error)/175,s=1)
    plt.hlines(y = threshold, xmin = 0, xmax = len(train_error),color = "red")

### Testing
def Testing(test, model, loss_function, defined_threshold,answer) :
    
    ### Error Point : 에러 점수 계산
    error_point = []
    for data in test :
        output = model(data)
        loss = loss_function(output, data)
        error_point.append(loss.item())

    ### Error Count : 예측한 에러의 갯수를 Counting
    
    treshold = np.array(defined_threshold)
    error = list(error_point >= treshold)

    ### Error Convert : True/False로 구분된 Error를 RECOVERING/NORMAL로 문자형 변환
    for i in range(len(error)) :
        if error[i] == True :
            error[i] = "RECOVERING"
        elif error[i] == False :
            error[i] = "NORMAL"
    
    ### Answer Cleaning : Answer 부분의 DataFrame의 index를 reset시키고, 필요없는 칼럼 제거
    answer = answer.reset_index()
    answer.drop(columns = ["index"], inplace = True)    

    ### Answer Sheet : 실제 Error와 예측한 Error의 차이를 보기 위한 데이터 프레임인 AnswerSheet 데이터 프레임 정의
    answer_sheet=pd.concat([pd.DataFrame(error),answer],axis = 1)
    
    ### Confusion Matrix
    metrix = metrics.confusion_matrix(answer_sheet["machine_status"], answer_sheet[0])
    
    ### Prediction Actual True
    true = metrix[0,0]/(metrix[0,0] + metrix[0,1])

    ### Prediction Actual False
    false = metrix[1,1]/(metrix[1,0] + metrix[1,1])

    ### Plotting : 예측한 결과 시각화(기존 데이터를 Threshold로 자른 Plot)
    print("Cutting Test Data Plot by Threshold")
    plt.figure(figsize = (10,2))
    plt.scatter(np.arange(0,len(error_point)),error_point,s=1)
    plt.hlines(y = defined_threshold, xmin = 0, xmax = len(error_point),color = "red")
    plt.show()
    print()
    print("Prediction Result")
    print("error : ",error.count(True))
    print("real_error: " ,answer.value_counts()["RECOVERING"])
    print("Threshold : ",treshold)
    print()
    print("Confusion Matrix")
    plt.plot(figsize = (10,10))
    sns.heatmap(metrix/np.sum(metrix), annot=True, fmt='.2%', cmap='Blues')
    print()
    print("Acutal True Predict Percentage : ", true)
    print("Acutal False Predict Percentage : ", false)




