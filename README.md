# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load dataset using pandas and split it into X (features) and y (target).
2.Split data into training and testing sets using train_test_split().
3.Train a Linear Regression model using LinearRegression().fit(X_train, y_train).
4.Predict values for X_test using regressor.predict(X_test).
5.Plot training and testing data with regression lines using matplotlib.
6.Calculate MSE, MAE, RMSE to evaluate model performance.
7.Improve model with feature scaling, different test splits, or polynomial regression. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PRIDEESH M
RegisterNumber:  212223040154
*/

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("House vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,Y_pred,color="blue")
plt.title("House vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/28daa515-f2dc-4a15-a237-2b88753c4121)
![image](https://github.com/user-attachments/assets/6e29baad-3b86-4797-86c1-812bc21d16c0)
![image](https://github.com/user-attachments/assets/fa8ca76a-0179-44d5-9e9d-f49d5955a78d)
![image](https://github.com/user-attachments/assets/6d5ff128-2287-4b9f-9ad2-3973a4b1e758)
![image](https://github.com/user-attachments/assets/7268f410-88ec-4309-b26e-ee5d4f3f59bb)
![image](https://github.com/user-attachments/assets/7268f410-88ec-4309-b26e-ee5d4f3f59bb)
![image](https://github.com/user-attachments/assets/e11ac01b-cf14-42a5-86fb-24c77fe1c20e)
![image](https://github.com/user-attachments/assets/a5a808b2-8e6b-495e-b6d0-742dea1137f9)
![image](https://github.com/user-attachments/assets/94eeb9b3-62a7-490a-a611-0a5bf058839d)
![image](https://github.com/user-attachments/assets/80f3e41e-9ca9-47d0-84b4-ddf85717c3ff)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
