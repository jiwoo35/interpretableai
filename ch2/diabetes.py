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

# correlation
corr = df_data.corr()

# visualization
sns.set(style='whitegrid')
sns.set_palette('bright')

f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(
	corr,
	vmin=-1, vmax=1, center=0,
	cmap="PiYG",
	square=True,
	ax=ax)
ax.set_xticklabels(
	ax.get_xticklabels(),
	rotation=90,
	horizontalalignment='right');
plt.show()
