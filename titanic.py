import pandas as pd

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Basic exploration
print(df.info())
print(df.describe())

# Value counts
print("Survived value counts:\n", df['Survived'].value_counts())

# Groupby
print("Survival rate by gender:\n", df.groupby('Sex')['Survived'].mean())

# Cleaning
print("Null values before cleaning:\n", df.isnull().sum())
df = df.dropna(subset=['Age'])  # Drop rows where Age is missing
df['Embarked'].fillna('S', inplace=True)  # Fill missing Embarked with 'S'

# Filtering
adults = df[df['Age'] > 18]
print("Adults in dataset:", adults.shape[0])