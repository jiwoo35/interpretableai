import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 회귀 분석을 위한 GAM 훈련에 사용할 pygam LinearGAM 클래스 import
from pygam import LinearGAM
# 수치 특성을 위한 평탄화 항 함수 import
from pygam import s
# 범주형 특성을 위한 요인 향 함수 import
from pygam import f

# data loading
diabetes=load_diabetes()
X, y = diabetes['data'], diabetes['target']

# feature rename
feature_rename={'age': 'Age',
				'sex': 'Sex',
				'bmi': 'BMI',
				'bp': 'BP',
				's1': 'Total Cholesterol',
				's2': 'LDL',
				's3': 'HDL',
				's4': 'Thyroid',
				's5': 'Glaucoma',
				's6': 'Glucose'}

df_data = pd.DataFrame(X, columns=diabetes['feature_names'])
df_data.rename(columns=feature_rename, inplace=True)
df_data['target']=y

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)

# 2.5 GAM
gam = LinearGAM(s(0)+ # 나이 특성을 위한 입방 스플라인 항
	f(1) + # 범주형 성별 특성을 위한 요인 항
	s(2) + # BMI 특성을 위한 입방 스플라인 항
	s(3) + # BP 
	s(4) + # 총 콜레스테롤 
	s(5) + # LDL
	s(6) + # HDL
	s(7) + # 갑상선
	s(8) + # 녹내장
	s(9), #혈당
	n_splines=35) # 특성별로 사용할 스플라인의 최대값

# 스플라인 개수, 매개변수 람다, 특성별 회귀 스플라인의 최적 가중치 등을 식별하기 위한 훈련과 교차 검증에 grid search 사용
gam.gridsearch(X_train, y_train)

y_pred=gam.predict(X_test)

mae=np.mean(np.abs(y_test-y_pred))

print(mae)

# GAM 해석 (1)
grid_locs1=[(0,0),(0,1),(1,0),(1,1)] # 2*2 4개 그래프 위치
fig, ax = plt.subplots(2, 2, figsize=(10,8)) # 2*2
selected_features = list(feature_rename.values())[:4] # 4가지 환자 메타데이터 특성 추출
for i, feature in enumerate(selected_features): # 반복 처리
	gl = grid_locs1[i]
	XX = gam.generate_X_grid(term=i) # 목표에 나머지 특성이 미치는 영향을 제외했을 때 특성 값에 대한 부분 의존성 생성
	ax[gl[0], gl[1]].plot(XX[:, i], gam.partial_dependence(term=i, X=XX)) # 부분 의존성 값을 실선으로 표시
	ax[gl[0], gl[1]].plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--') # 부분 의존성 값을 기준으로 95% 신뢰도 구간을 점선으로 표시
	ax[gl[0], gl[1]].set_xlabel('%s' % feature)
	ax[gl[0], gl[1]].set_ylabel('f(%s)' % feature)
	
#plt.show()

# GAM 해석 (2)
grid_locs2=[(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)] # 3*2 6개 그래프 위치
fig2, ax2 = plt.subplots(3, 2, figsize=(12,12)) # 3*2
selected_features2 = list(feature_rename.values())[4:] # 6가지 환자 메타데이터 특성 추출

for i, feature in enumerate(selected_features2): # 반복 처리
	idx = i+4
	gl = grid_locs2[i]
	XX = gam.generate_X_grid(term=idx) # 목표에 나머지 특성이 미치는 영향을 제외했을 때 특성 값에 대한 부분 의존성 생성
	ax2[gl[0], gl[1]].plot(XX[:, idx], gam.partial_dependence(term=idx, X=XX)) # 부분 의존성 값을 실선으로 표시
	ax2[gl[0], gl[1]].plot(XX[:, idx], gam.partial_dependence(term=idx, X=XX, width=.95)[1], c='r', ls='--') # 부분 의존성 값을 기준으로 95% 신뢰도 구간을 점선으로 표시
	ax2[gl[0], gl[1]].set_xlabel('%s' % feature)
	ax2[gl[0], gl[1]].set_ylabel('f(%s)' % feature)

plt.show()