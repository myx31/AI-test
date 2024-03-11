import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns

data_df = pd.read_csv('wdbc.csv')
#data_df.info()
print(data_df.head())
# Xác định giá trị lớn nhất và nhỏ nhất trong cột 'A'
max_value_A = data_df['radius1'].max()
min_value_A = data_df['radius1'].min()

# Xác định giá trị lớn nhất và nhỏ nhất trong cột 'B'
max_value_B = data_df['radius2'].max()
min_value_B = data_df['radius2'].min()

print("Giá trị lớn nhất của cột 'radius1':", max_value_A)
print("Giá trị nhỏ nhất của cột 'radius1':", min_value_A)
print("Giá trị lớn nhất của cột 'radius2':", max_value_B)
print("Giá trị nhỏ nhất của cột 'radius2':", min_value_B)

