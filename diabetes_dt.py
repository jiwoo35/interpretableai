from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
# 이진트리 생성 및 시각화에 필요한 libraries
from io import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.environ['PATH'] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin'

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

# 2.4 Decision Tree
# 결정 트리 회귀 초기화, 지속적이고 재현 가능한 결과를 얻기 위해 random_state 설정
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

# MAE 측정 지표로 모델 성능 평가
mae=np.mean(np.abs(y_test-y_pred))

print(mae)

# Visualization
# DOT 포맷으로 이진 트리/그래프 저장하기 위해 string buffer 초기화
diabetes_dt_dot_data=StringIO()
# DOT 포맷 이진 트리로 결정 트리 모델 추출
export_graphviz(dt_model, out_file=diabetes_dt_dot_data, 
				filled=False, rounded=True,	feature_names=diabetes['feature_names'],
				proportion=True, precision=1, special_characters=True)
# DOT 포맷 String 사용해서 이진 트리 시각화 생성
dt_graph = pydotplus.graph_from_dot_data(diabetes_dt_dot_data.getvalue())
# Image 클래스 사용해서 이진 트리 시각화
Image(dt_graph.create_png())
#dt_graph.create_png()
# PNG 파일로 저장
dt_graph.write_png("decision_tree.png")

# 특성 중요도 추출, 시각화
# 훈련된 결정 트리 모델에서 특성 중요도 추출
weights = dt_model.feature_importances_

# 특성 가중치 인덱스를 낮은 것부터 정렬
feature_importance_idx = np.argsort(np.abs(weights))[::-1]

# 중요도가 낮은 것부터 특성명과 특성 가중치 추출
feature_importance = [df_data.columns[idx].upper() for idx in feature_importance_idx]
feature_importance_values = [weights[idx] for idx in feature_importance_idx]

# 시각화
f, ax = plt.subplots(figsize=(10,8))
sns.barplot(x=feature_importance_values, y=feature_importance, ax=ax)
ax.grid(True)
ax.set_xlabel('Feature Weights')
ax.set_ylabel('Features')

plt.show()