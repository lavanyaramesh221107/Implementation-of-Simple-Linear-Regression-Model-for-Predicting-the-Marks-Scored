# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
~~~
1. Load the dataset into a DataFrame and explore its contents to understand the data structure.
2.Separate the dataset into independent (X) and dependent (Y) variables, and split them into training and testing sets.
3.Create a linear regression model and fit it using the training data.
4.Predict the results for the testing set and plot the training and testing sets with fitted lines.
5.Calculate error metrics (MSE, MAE, RMSE) to evaluate the model’s performance.
~~~

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Lavanya R 
RegisterNumber:212225230149
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset (Study Hours vs Marks)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([35, 50, 65, 70, 85])

# Create model
model = LinearRegression()

# Train model
model.fit(X, Y)

# Predict marks
Y_pred = model.predict(X)

# Display slope and intercept
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# Plot graph
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y_pred, color='red', label='Regression Line')
plt.xlabel("Study Hours")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression - Marks Prediction")
plt.legend()
plt.show()
```

## Output:
<img width="1105" height="640" alt="image" src="https://github.com/user-attachments/assets/5e470ba5-327b-4d93-abbf-6272b9c66ace" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
