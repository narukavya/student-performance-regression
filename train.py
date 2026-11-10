import pandas as pd

df = pd.read_csv(
    "data/student-mat.csv",
    sep=";"
)

print(df.shape)

# print(df.head())

print(df.isnull().sum())