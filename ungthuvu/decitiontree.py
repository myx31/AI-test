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
X= data_df.iloc[:,2:32].values
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


#trainning  Ml model 1

from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(X_train,Y_train)

# Dự đoán trên tập kiểm tra
Y_preds = dt_model.predict(X_test)


#print(Y_preds)
print(pd.DataFrame({'Y':Y_test, 'Y_preds':Y_preds}))



# Đánh giá hiệu suất bằng cross-validation
from sklearn.model_selection import cross_val_score
# Thực hiện xác thực chéo
# tham số cv xác định số lần gấp
# tham số tính điểm chỉ định số liệu sẽ sử dụng
# Ở đây chúng tôi sử dụng độ chính xác làm thước đo tính điểm
scores = cross_val_score(dt_model ,X, Y, cv=10)

# In điểm chính xác cho mỗi lần gấp
print("Cross-Validation Scores:", scores)
# In điểm chính xác trung bình
print("Mean CV Score:", scores.mean())


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_preds)
print("Accuracy:", accuracy)

# Tính toán độ chính xác
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_preds)
print("Precision:", precision)

# Tính toán điểm F1
from sklearn.metrics import f1_score
f1 = f1_score(Y_test, Y_preds)
print("F1 Score:", f1)

# Recall
from sklearn.metrics import recall_score
recall = recall_score(Y_test, Y_preds)
print("Recall:", recall)

x_real1 = 13.96,17.05,91.43,602.4,0.1096,0.1279,0.09789,0.05246,0.1908,0.0613,0.425,0.8098,2.563,35.74,0.006351,0.02679,0.03119,0.01342,0.02062,0.002695,16.39,22.07,108.1,826,0.1512,0.3262,0.3209,0.1374,0.3068,0.07957
y_pred1 = dt_model.predict([x_real1])
print("Giá trị y dự đoán:", y_pred1)

x_real2 = 10.6,18.95,69.28,346.4,0.09688,0.1147,0.06387,0.02642,0.1922,0.06491,0.4505,1.197,3.43,27.1,0.00747,0.03581,0.03354,0.01365,0.03504,0.003318,11.88,22.94,78.28,424.8,0.1213,0.2515,0.1916,0.07926,0.294,0.07587
y_pred2 = dt_model.predict([x_real2])
print("Giá trị y dự đoán:", y_pred2)

x_real3 = 1.87,1.21,8.38,512.2,0.09425,0.0219,0.039,0.01615,0.01,0.0569,0.2345,0.219,1.546,18.24,0.005518,0.02178,0.02589,0.00633,0.02593,0.002157,3.9,23.64,9.7,597.5,0.1256,0.1808,0.1992,0.0578,0.3604,0.07062
y_pred3 = dt_model.predict([x_real3])
print("Giá trị y dự đoán:", y_pred3)