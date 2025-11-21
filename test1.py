import pandas as pd
df=pd.read_csv('bank_transactions.csv')
df=df[:5000]
df.to_csv('data.csv',index=False)