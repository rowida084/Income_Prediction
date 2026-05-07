import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("train_data.csv")

sns.set_palette("Greens_r")
sns.set_style("whitegrid")

df.columns=df.columns.str.strip()

sns.histplot(df['age'], kde=True)
plt.title("Age Distribution")
plt.show()

sns.countplot(data=df, x='sex', hue='Income')
plt.title("Income Distribution by Gender")
plt.show()

sns.boxplot(data=df, x='Income', y='age')
plt.title("Age vs Income")
plt.show()

sns.boxplot(data=df, x='Income', y='hours-per-week')
plt.title("Working Hours vs Income")
plt.show()

sns.histplot(df['capital-gain'], bins=50)
plt.title("Capital Gain Distribution")
plt.show()

sns.histplot(df['capital-loss'], bins=50)
plt.title("Capital Loss Distribution")
plt.show()

edu_counts = pd.crosstab(df['education'], df['Income'])
edu_pct = edu_counts.div(edu_counts.sum(1), axis=0)
edu_pct.plot(kind='bar', stacked=True)
plt.title("Proportion of Income by Education")
plt.xticks(rotation=45)
plt.show()

sns.countplot(data=df, x='marital-status', hue='Income')
plt.xticks(rotation=45)
plt.title("Marital Status vs Income")
plt.show()

sns.countplot(data=df, x='occupation', hue='Income')
plt.xticks(rotation=45)
plt.title("Occupation vs Income")
plt.show()

sns.countplot(data=df, x='workclass', hue='Income')
plt.xticks(rotation=45)
plt.title("Workclass vs Income")
plt.show()

important_cols = ['age', 'hours-per-week', 'education-num',
                  'capital-gain', 'capital-loss']
corr = df[important_cols].corr()
sns.heatmap(corr, annot=True, cmap='Greens_r')
plt.title("Correlation Between Features")
plt.show()
