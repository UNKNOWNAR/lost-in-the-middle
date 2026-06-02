import pandas as pd

df = pd.read_parquet("./data/pair/train-00000-of-00001.parquet")
print(df.iloc[0]['query'])