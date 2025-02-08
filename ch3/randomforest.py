import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from pdpbox import pdp, info_plots
from pdpbox.pdp import PDPIsolate

# data loading
df=pd.read_csv('StudentsPerformance.csv')

# 입력 특성 인코딩
gender_le=LabelEncoder()
race_le=LabelEncoder()
parent_le=LabelEncoder()
lunch_le=LabelEncoder()
test_prep_le=LabelEncoder()
df['gender_le']=gender_le.fit_transform(df['gender'])
df['race_le']=race_le.fit_transform(df['race/ethnicity'])
df['parent_le']=parent_le.fit_transform(df['parental level of education'])
df['lunch_le']=lunch_le.fit_transform(df['lunch'])
df['test_prep_le']=test_prep_le.fit_transform(df['test preparation course'])

# 목표 변수 인코딩
math_grade_le=LabelEncoder()
reading_grade_le=LabelEncoder()
writing_grade_le=LabelEncoder()
df['math_grade_le']=math_grade_le.fit_transform(df['math score'])
df['reading_grade_le']=reading_grade_le.fit_transform(df['reading score'])
df['writing_grade_le']=writing_grade_le.fit_transform(df['writing score'])

# 훈련/검증/테스트 세트 생성
df_train_val, df_test = train_test_split(df, test_size=0.2)
stratify = df['math_grade_le'], #F shuffle=True, random_state=42)
feature_cols = ['gender_le', 'race_le', 'parent_le', 'lunch_le', 'test_prep_le']
X_train_val = df_train_val[feature_cols]
X_test = df_test[feature_cols]
y_math_train_val = df_train_val['math_grade_le']
y_reading_train_val = df_train_val['reading_grade_le']
y_writing_train_val = df_train_val['writing_grade_le']
y_math_test = df_test['math_grade_le']
y_reading_test = df_test['reading_grade_le']
y_writing_test = df_test['writing_grade_le']

def create_random_forest_model(n_estimators, # 랜덤 포레스트에 포함할 결정트리 개수 설정
	max_depth=10, # 결정 트리의 최대 계층 매개변수
	criterion='gini', # 각 결정 트리를 최적화하기 위한 손실 함수로 지니 불순도를 사용
	random_state=42, n_jobs=4): # n_jobs: 컴퓨터의 모든 가용한 코어를 사용해서 개별 결정 트리를 병렬로 훈련
	return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, random_state=random_state, n_jobs=n_jobs)

# 랜덤 포레스트 모델 초기화 및 훈련
math_model = create_random_forest_model(50) # 수학 과목에 대한 랜덤 포레스트 분류 모델을 결정 트리 50개로 초기화
math_model.fit(X_train_val, y_math_train_val)
y_math_model_test = math_model.predict(X_test)
print("math training is finished")

reading_model = create_random_forest_model(25)
reading_model.fit(X_train_val, y_reading_train_val)
y_reading_model_test = reading_model.predict(X_test)
print("reading training is finished")

writing_model = create_random_forest_model(40)
writing_model.fit(X_train_val, y_writing_train_val)
y_writing_model_test = writing_model.predict(X_test)
print("writing training is finished")

# 3.3 특성 중요도 정규화
math_fi = math_model.feature_importances_ * 100 # 수학 랜덤포레스트 모델에서 특성 중요도 추출
reading_fi = reading_model.feature_importances_ * 100
writing_fi = writing_model.feature_importances_ * 100

feature_names=['Gender', 'Ethinicity', 'Parent Level of Education', 'Lunch', 'Test Preparation']

# 시각화
fig, ax = plt.subplots()
index = np.arange(len(feature_names))
bar_width = 0.2
opacity = 0.9
error_config = {'ecolor':'0.3'}
ax.bar(index, math_fi, bar_width, alpha=opacity, color='r', label='Math Grade Model')
ax.bar(index+bar_width, reading_fi, bar_width, alpha=opacity, color='g', label='Reading Grade Model')
ax.bar(index+bar_width*2, writing_fi, bar_width, alpha=opacity, color='b', label='Writing Grade Model')
ax.set_xlabel('')
ax.set_ylabel('Feature Importance(%)')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(feature_names)
for tick in ax.get_xticklabels():
	tick.set_rotation(90)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True)
# plt.show();

# 3.4 model agnostic
feature_cols = ['gender_le', 'race_le', 'parent_le', 'lunch_le', 'test_prep_le']
#print(dir(pdp))
#help(PDPIsolate)
pdp_education = PDPIsolate(model=math_model, # 훈련된 수학 랜덤포레스트 모델을 불러와서 각 교육 수준별 부분의존성함수(PDP) 도출
	df=df, model_features=feature_cols, feature='parent_le', feature_name='Parent Level Education')
ple_xticklabels=['High School', 'Some High School', 'Some College', 'Associate\'s Degree', 'Bachelor\'s Degree', 'Master\'s Degree']

# PDP 도표를 위한 parameter
plot_params = {
	# plot title and subtitle
	'title': 'PDP for Parent Level Educations - Math Grade',
	'subtitle': 'Parent Level Education(legend): \n%s' %(ple_xticklabels),
	'title_fontsize': 15,
	'subtitle_fontsize': 12,
	# line color
	'contour_color': 'white',
	'font_family': 'Arial',
	# matplotlib color map for interaction map
	'cmap': 'viridis',
	# alpha color for interaction map
	'inter_fill_alpha': 0.8,
	# font size for interaction map
	'inter_fontsize': 9,
}

# PDP for Parent Education Level in matplotlib
# pdp_plot 에러
#fig, axes = pdp.pdp_plot(pdp_isolate_out=pdp_education, 
#	feature_name='Parent Level Education', center=True, x_quantile=False, ncols=2, 
#	plot_lines=False, frac_to_plot=100, plot_params=plot_params, figsize=(18,25))

# Pdpbox API Reference: https://pdpbox.readthedocs.io/en/latest/PDPIsolate.html
fig, axes = pdp_education.plot(center=True, show_percentile=False, ncols=2,
	plot_lines=False, frac_to_plot=100, plot_params=plot_params, engine="matplotlib", figsize=(18,25))
# engine 설정을 안하면, 기본적으로 ploltly으로 작동하여 matplotlib로 설정해줌
# 그랬더니 아래와 같이 또 에러;
# _tkinter.TclError: not enough free memory for image buffer
axes['pdp_ax'][0].set_xlabel('Parent Level Education')
axes['pdp_ax'][1].set_xlabel('Parent Level Education')
axes['pdp_ax'][2].set_xlabel('Parent Level Education')
axes['pdp_ax'][3].set_xlabel('Parent Level Education')
axes['pdp_ax'][0].set_title('Grade A')
axes['pdp_ax'][1].set_title('Grade B')
axes['pdp_ax'][2].set_title('Grade C')
axes['pdp_ax'][3].set_title('Grade F')
axes['pdp_ax'][0].set_xticks(parent_codes)
axes['pdp_ax'][1].set_xticks(parent_codes)
axes['pdp_ax'][2].set_xticks(parent_codes)
axes['pdp_ax'][3].set_xticks(parent_codes)
axes['pdp_ax'][0].set_xticklabels(ple_xticklabels)
axes['pdp_ax'][1].set_xticklabels(ple_xticklabels)
axes['pdp_ax'][2].set_xticklabels(ple_xticklabels)
axes['pdp_ax'][3].set_xticklabels(ple_xticklabels)
for tick in axes['pdp_ax'][0].get_xticklabels():
	tick.set_rotation(45)
for tick in axes['pdp_ax'][1].get_xticklabels():
	tick.set_rotation(45)
for tick in axes['pdp_ax'][2].get_xticklabels():
	tick.set_rotation(45)
for tick in axes['pdp_ax'][3].get_xticklabels():
	tick.set_rotation(45)