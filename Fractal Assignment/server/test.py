import pandas as pd



csv = 'employee_data.csv'
df = pd.read_csv(csv)
result = df.groupby('Location').count()['Employee_Code']

for index, row in result.iterrows():
    print(row)