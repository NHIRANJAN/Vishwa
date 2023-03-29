#Student marks prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection  import  train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import time
from scipy.stats import linregress

#Loading the data Sets
df=pd.read_csv("student_info.csv")
print(df.head())


#EDA of the Datasets
print(df.shape)
print(df.describe())


#Data cleaning as we have missing values in data.

print(df.isnull().sum())

y=df['study_hours'].mean()

df2=df.fillna(y)

print(df2.isnull().sum())

#Visualization of the datasets.
plt.scatter(x=df2.study_hours,y=df.student_marks)
plt.xlabel("Student study hours")
plt.ylabel("Student marks")
plt.title("Student marks vs prediction")
plt.show()

#Spilting the data sets.
X=df2.drop("student_marks", axis='columns')
y=df2.drop("study_hours", axis='columns')

print("Shape of the X=" ,X.shape)
print("Shape of the Y=" ,y.shape)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=5)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.linear_model import  LinearRegression

lin = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=5)
rfr = RandomForestRegressor()

print(lin.fit(X_train,y_train))
print(knn.fit(X_train,y_train))
print(rfr.fit(X_train,y_train.values.ravel()))
#
# print(linear.coef_)
# print(linear.intercept_)
y_pred=lin.predict((X_test))
yy=y_pred.round(2)

df3=pd.DataFrame(np.c_[X_test,y_test,yy], columns=['student_int_marks', 'student_original_marks','Student_predicted_marks'])

print(df3)

y_pred1=rfr.predict((X_test))
yy1=y_pred1.round(2)

df4=pd.DataFrame(np.c_[X_test,y_test,yy1], columns=['student_int_marks', 'student_original_marks','Student_predicted_marks'])

y_pred2=knn.predict((X_test))
yy2=y_pred2.round(2)

df5=pd.DataFrame(np.c_[X_test,y_test,yy2], columns=['student_int_marks', 'student_original_marks','Student_predicted_marks'])


#Fine tune your model.

print("Lin_Score:",lin.score(X_test,y_test))
print("Knn_Score:",knn.score(X_train,y_train))
print("rfr_Score:",rfr.score(X_train,y_train))

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
import sklearn.metrics as metrics
r2_lin = metrics.r2_score(X_test,y_pred)
print("Lin_R-Squared:", r2_lin)
r2_knn = metrics.r2_score(X_test,y_pred1)
print("Knn_R-Squared:", r2_knn)
r2_rfr = metrics.r2_score(X_test,y_pred2)
print("Rfr_R-Squared:", r2_rfr)

def mape(y_test, pred):
    y_test, pred = np.array(y_test), np.array(pred)
    mape = np.mean(np.abs((y_test - pred) / y_test))
    value=round(mape*100,2)
    return value
print("Lin_MAPE:",mape(y_test,yy))
print("Knn_MAPE:",mape(y_test,yy1))
print("Rfr_MAPE:",mape(y_test,yy2))

#

plt.scatter(X_test,y_test)
plt.plot(X_train,lin.predict(X_train),color="r" )
plt.show()

plt.scatter(X_test,y_test)
plt.plot(X_train,knn.predict(X_train),color="r" )
plt.show()

plt.scatter(X_test,y_test)
plt.plot(X_train,rfr.predict(X_train),color="r" )
plt.show()

#Saving Our Model.

import joblib
joblib.dump(lin,'main.pkl')

