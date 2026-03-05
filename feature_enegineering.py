### Inspecting the dataset

# We start by importing the libraries and the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('healthcare-dataset-stroke-data.csv')


# Following that, the shape, information and dataset will be shown.
print("Dataset shape:", df.shape)  
print(df.head())  

print("\nInfo:\n")
df.info()  


#### Stroke distribution

df['stroke'].value_counts()
df['stroke'].value_counts(normalize=True)
sns.countplot(x='stroke', data=df)
plt.title("Stroke Distribution")
plt.show()
# The dataset is highly imbalanced, as only 5% of people had a stroke. This will guide modeling decisions, such as using class weights or evaluation metrics beyond accuracy.

sns.boxplot(x='stroke', y='age', data=df)
df.groupby('stroke')['age'].mean()
sns.boxplot(x='stroke', y='avg_glucose_level', data=df)
sns.boxplot(x='stroke', y='bmi', data=df)

# People who had a stroke tend to have higher average glucose levels and slightly higher BMI.


#### Missing values

# Let's now check for missing values.
df.isnull().sum()

# As we saw, there are 201 missing values in the ``bmi`` column, so let's fill them using the median.
df.fillna({'bmi': df['bmi'].median()}, inplace=True)
df.isnull().sum()
# There are no missing values now.


#### Categorical values

# Now we inspect the unique values for each categorical feature in order to help us get a better view at them.
for col in df.select_dtypes(include='object'):
    print(col, df[col].unique())

# There's at least 1 person with smoking status as 'Unknown'. The rest seems correct, although there is an extra gender that does not exist whatsoever: let's see that.
num_other_gender_rows = df[df['gender'] == 'Other'].shape[0]
print('Total other gender rows:', num_other_gender_rows)

# It's really just one person, so let's just discover the index in order to remove that row.
df[df['gender'] == 'Other']
df.drop(3116, inplace = True)
print('Genders:', df['gender'].unique())


# Next, we check how many people have an unknown smoking status.
num_unk_smoke = df[df['smoking_status'] == 'Unknown'].shape[0]
print('Total number of people with unknown people smoking status:', num_unk_smoke)

# 1544 people have an unknown smoking status, that is 30% of the whole dataset.
# We'll need to find an effective way to determine these statuses.


#### Numerical values