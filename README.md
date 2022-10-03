# 주성분 분석을 이용한 AutoEncoder기반 이상탐지 모델 성능 개선

![슬라이드1](https://user-images.githubusercontent.com/89781598/193517912-0e3bf350-91ed-4b56-9686-863f7dadb538.JPG)

## 개요
* 본 프로젝트는 포스코 스틸브릿지 설비기술 직무 기술면접에 사용되었습니다.
* 주성분 분석을 사용하면 이상 탐지 모델의 성능이 개선된다는 것을 이야기하고 싶었습니다.

## 데이터
* 데이터 셋은 아래 링크를 통해 다운로드 받을 수 있습니다.
```
https://www.kaggle.com/datasets/nphantawee/pump-sensor-data
```

## 개발환경
<p align="center">
  <img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white"/></a>&nbsp
  <br>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white"/></a>&nbsp
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=NumPy&logoColor=white"/></a>&nbsp
  <br>
    <img src="https://img.shields.io/badge/-matplotlib-blue"/></a>&nbsp
</p>

## 파일 경로
```
📦POSCO.PumpSensor
 ┣ 📂Package
 ┃ ┣ 📜Data.py
 ┃ ┣ 📜Modeling.py
 ┃ ┗ 📜PCA.py
 ┗ 📂Result
 ┃ ┣ 📜After.ipynb
 ┃ ┗ 📜Before.ipynb
```

## 파일
- Package : Results를 도출하기 위한 함수들이 들어있는 패키지입니다.
    - Data.py : 데이터를 전처리하는 함수, 센서 각각을 explore하는 함수, interploate 전과 후를 비교하는 함수가 들어있음.
    - Modeling.py : 오토인코더 모델 정의. 훈련, 임계값 정의, 테스트 함수가 들어있음.
    - PCA.py : PCA 수행함수, biplot plotting 함수가 들어있음.
- Result : PCA를 하기 전과 한 후에 대한 코드가 들어있습니다.
    - After.ipynb : PCA 수행 후 결과를 볼 수 있음
    - Before.ipynb : PCA 수행 전 결과를 볼 수 있음
    
## EDA와 without PCA modeling
![슬라이드2](https://user-images.githubusercontent.com/89781598/193519137-55fec0ee-f9f3-4b70-95b5-e6d70d91a928.JPG)

## PCA and Demension Reduction
![슬라이드3](https://user-images.githubusercontent.com/89781598/193519189-5be57c4c-8f60-43ab-bfd4-4f401a95f602.JPG)

## With PCA Result
![슬라이드4](https://user-images.githubusercontent.com/89781598/193519204-03d9055a-5021-40d3-a9a5-0fc4b470c565.JPG)

## 문의사항
* email : ajc227ung@gmail.com
