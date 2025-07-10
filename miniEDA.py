import pandas as pd

# Load the Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Print basic insights
print("Dataset Info:\n")
print(df.info())

print("\nBasic Stats:\n")
print(df.describe())

print("\nNull Values:\n")
print(df.isnull().sum())

print("\nData Types:\n")
print(df.dtypes)