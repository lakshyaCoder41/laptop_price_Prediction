import pandas as pd
data = pd.read_excel(r'newdata.xlsx')
print(data.columns[6:-1])
