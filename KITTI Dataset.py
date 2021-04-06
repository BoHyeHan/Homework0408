#!/usr/bin/env python
# coding: utf-8

# # 데이터 셋 
# 
# ## 1. KITTY Dataset

# **움직이는 자동차에 카메라를 달고 운행을 하면서 거리의 영상을 깊이 값과 함께 담아낸 데이터셋이다.**
# 
# 모바일 로봇 공학 및 자율 주행에 사용되는 가장 인기있는 데이터 세트 중 하나이며, 고해상도 RGB, 그레이 스케일 스테레오 카메라 및 3D 레이저 스캐너를 포함하여 다양한 센서 양식으로 기록 된 몇 시간의 교통 시나리오로 구성된다. 
# 
# **1.구성** <br>
# 
#   총 7,481개의 시퀀스 학습 데이터 (9가지의 객체 종류, 51,867개의 라벨로 이루어짐)
#   
#   - test (711)
#   - train (6,347)
#   - validation (423) 
#   
#   a) Data Description  <br>
#   
#      : 센서가 기록한 모든 시퀀스 데이터  <br>
#      
#      ** `Image` : color 와 grayscale 이미지들이 8-bit PNG 파일로 저장되어 있다. <br>
#      ** `OXTS(GPS/IMU)` : 각각의 프레임에서, 30개의 다른 GPS/IMU 값이 text file에 저장한다.<br>
#      ** `Velodyne`: 효율성을 위해, 분석이 용이한 부동 소수점 바이너리로 velodyne scan이 point를 저장한다.(포인트: (x,y,z)좌표, 반사율 값 r) <br> 
#      
#   b) Annotations  <br>
#   
#      : 'car','van','truck','pedestrian','person(sitting)','cyclist','tram','misc'의 8개의 클래스가 나누어져 있다.  <br>

# **2.데이터 탐색**
# 
# * 데이터셋을 사용하기 위해 먼저 설치가 필요하다.<br> 
# ```python
# !pip install pykitti 
# ``` 
# 
# * Data Example <br>
# 
# KITTI Dataset 홈페이지에서 2011_09_26_drive_0001(0.4GB)의 영상을 가져와 살펴보면<br>
# 
#   1) `Length` : 114 frame (00:11 minutes)<br>
#   2) `Image resolution` : 1392 x 512 pixels<br>
#   3) `Labels` : 12 Cars ,0 Vans, 0 Trucks, 0 Pedestrians, 0 Sitters, 2 Cyclists, 1 Trams, 0 Misc<br>
#   
#       
# * 데이터셋 시각화 (point cloud의 약 20% 만 시각화 하며, 각 프레임은 120k points를 포함한다.)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# **3.데이터 셋의 활용** <br>
# 
# `딥러닝 기반 차량감지`, `라이터 포인트 투영`, `자율주행 자동차간 거리 측정` 등 다양한 자율주행 기술연구에 사용되고 있다. 
