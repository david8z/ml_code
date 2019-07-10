# Categorical Features: Sex, Embarked
# Ordinal Features: PClass
# Continuous Features: Age

import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt



# Get titanic data as a DataFrame
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# print(train_df.head())
# print(train_df.info())


# Remove non-relevant data
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)
test_df = test_df.drop(['Name', 'Ticket'], axis = 1)

train_df["Embarked"] = train_df["Embarked"].fillna("S")
print(train_df.info())

sns.factorplot('Embarked', 'Survived', data=train_df, height=4, aspect=3)
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
# sns.factorplot('Embarked',data=train_df,kind='count',order=['S','C','Q'],ax=axis1)
# sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)
sns.countplot(x='Embarked', data=train_df, ax=axis1)
sns.countplot(x='Survived', hue="Embarked", data=train_df, order=[1,0], ax=axis2)

# group by embarked, and get the mean for survived passengers for each value in Embarked
embark_perc = train_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)
plt.show()
