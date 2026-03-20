import pandas as pd

df = pd.read_csv("artifacts/train.csv")
print(df.iloc[:, -1].value_counts())