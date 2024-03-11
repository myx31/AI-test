import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
data_df = pd.read_csv('wdbc.csv')

#print(data_df.head())
#print(data_df.columns)

# missing data

#data_df.info()
#for col in data_df.columns:

#    missing_data = data_df[col].isna().sum()
#    missing_percent = missing_data/len(data_df)*100
#    print(f"Column {col}: has {missing_percent}%")

#fig, ax = plt.subplots(figsize=(10,8))
#sns.heatmap(data_df.isna(), cmap="Blues", cbar= False, yticklabels=False);
#plt.show()

# tách giá trị X,Y
X= data_df.iloc[:,2:31].values
#print(X)

Y= data_df.iloc[:,1]
#print(Y)


# thay thế giá trị bị miss
#from sklearn.impute import SimpleImputer
#imputer=SimpleImputer(missing_values=np.nan, strategy='mean') #mean là gia trị TB của côt
#imputer.fit(X[:,1:3])
#X[:,1:3] = imputer.transform(X[:, 1:3])

# mã hóa X
#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder

#ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder() , [0])], remainder='passthrough')
#X= ct.fit_transform(X)
#print(X)

# mã hóa Y
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
Y= le.fit_transform(Y)
#print(Y) # 1 ác tính, 0 là lành tính


# cách tách dữ liệu thành training và test sets
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=0)

# feature scaling làm cho các giá trị chênh lệch có cung thang co dãn dữ liêu

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train [:,:] = sc.fit_transform(X_train[:,:])
#print(X_train.shape)
#print(X_test.shape)

X_test [:,:] = sc.fit_transform(X_test[:,:])
#print(X_test)

#trainning  Ml model 1

from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(X_train,Y_train)

Y_preds = dt_model.predict(X_test)
#print(Y_preds)

print(pd.DataFrame({'Y':Y_test, 'Y_preds':Y_preds}))

#trainning Ml model 2

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

rf_model = RandomForestRegressor ( random_state=1)
rf_model.fit(X_train,Y_train)

rf_preds = rf_model.predict(X_test)
#print(rf_preds)

# dự đoán 1 TH
#print(X_test.shape) phải giảm feature
#rf_model.predict()
# đánh giá




