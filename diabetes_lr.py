from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sns

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

# 2.3 Linear Regression
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

lr_model=LinearRegression()

lr_model.fit(X_train, y_train)

y_pred=lr_model.predict(X_test)

mae=np.mean(np.abs(y_test-y_pred))

print(mae)

# Visualization
sns.set(style='whitegrid')
sns.set_palette('bright')

# coef_parameter를 사용해서 앞에서 훈련한 linear regression model의 weight 추출
weights=lr_model.coef_

# 가중치를 중요도가 낮은 것부터 정렬해서 인덱스 획득
feature_importance_idx=np.argsort(np.abs(weights))[::-1]

# 나열된 인덱스를 갖고 feature names와 연관 가중치 획득
feature_importance=[diabetes['feature_names'][idx].upper() for idx in feature_importance_idx]
feature_importance_values=[weights[idx] for idx in feature_importance_idx]

# graph
f, ax = plt.subplots(figsize=(10,8))
sns.barplot(x=feature_importance_values, y=feature_importance, ax=ax)
ax.grid(True)
ax.set_xlabel('Feature Weights')
ax.set_ylabel('Features')

plt.show()
