from google.colab import files
uploaded=files.upload()


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
sns.set_palette("Greens_r")

df = pd.read_csv('processed_test_hott_data.csv')
df.head()

sns.histplot(df['age'], kde=True)

plt.title("Age Distribution", fontweight='bold')
plt.show()

sns.boxplot( data=df,x='Income', y='age',)

plt.title("Age vs Income", fontweight='bold')
plt.show()


sns.countplot(data=df, x='Income')
plt.title("Income Class Distribution")
plt.show()


df['capital-gain'] = pd.to_numeric(df['capital-gain'], errors='coerce')
df['has_capital_gain'] = (df['capital-gain'] > 0).astype(int)
df['has_capital_loss'] = (df['capital-loss'] > 0).astype(int)
df['capital_gain_log'] = np.where(
    df['capital-gain'] > 0,
    np.log1p(df['capital-gain']),
    0)

df['capital_loss_log'] = np.where(
    df['capital-loss'] > 0,
    np.log1p(df['capital-loss']),
    0)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df[df['capital_gain_log'] > 0]['capital_gain_log'], bins=50, ax=axes[0])
axes[0].set_title("Capital Gain (Log, non-zero only)")

sns.histplot(df[df['capital_loss_log'] > 0]['capital_loss_log'], bins=50, ax=axes[1])
axes[1].set_title("Capital Loss (Log, non-zero only)")
plt.show()

sns.boxplot(x='Income', y='hours-per-week', data=df,)

plt.title("Working Hours vs Income", fontweight='bold')
plt.show()

edu_income_counts=pd.crosstab(df['education-num'].round(2),df['Income'])
edu_income_pct=edu_income_counts.div(edu_income_counts.sum(1),axis=0)
edu_income_pct.plot(kind='bar',stacked=True)


plt.title("Proportion of income Levels by Education", fontweight='bold')
plt.show()

ax = sns.countplot(data=df, x='sex', hue='Income')

ax.set_xticklabels(['Female', 'Male'])

plt.title("Income Distribution by Gender", fontsize=14, fontweight='bold')
plt.xlabel("Gender", fontweight='bold')
plt.ylabel("Count", fontweight='bold')

plt.legend(title='Income', labels=['<=50K', '>50K'])

plt.show()


important_cols = [
    'age',
    'hours-per-week',
    'education-num',
    'capital-gain',
    'capital-loss',
    'Income'
]

corr = df[important_cols].corr()

sns.heatmap(corr, annot=True, cmap = 'Greens_r')

plt.title("Correlation Between Features", fontweight='bold')
plt.show()


