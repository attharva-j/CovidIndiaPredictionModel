#   Polynomial Regression

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from activeCasesPrediction import predictActiveCases
from recoveredCasesPrediction import predictRecoveredCases
from deathCasesPrediction import predictDeathCases

# Importing the dataset
dataset = pd.read_csv('covid.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""


#   Fitting Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#   Fitting Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)


# Predicting the Test set results
y_pred = lin_reg2.predict(poly_reg.transform(X_test))
np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
print("accuracy = " + str(r2_score(y_test, y_pred)*100) + "%")

day = int(input("Input a day to predict case count:"))
print("Confirmed case prediction count for 18-05-2020: " + str(int(lin_reg2.predict(poly_reg.fit_transform([[day]]))[0][0])))

from xlwt import Workbook

# Workbook is created
wb = Workbook()

# add_sheet is used to create sheet.
sheet1 = wb.add_sheet('covid')
row = 1
column = 7

for i in range(1,201):
    sheet1.write(row, column, int(lin_reg2.predict(poly_reg.fit_transform([[i]]))[0][0]))
    row+=1
wb.save('C:/Users/Atharva Joshi/Desktop/covidDemo.xls')

"""
#   Visualizing the linear regression
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Salary Prediction(Linear Reg.)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
"""

#   Visualizing the polynomial regression
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Covid-19(India) Confirmed Case Prediction(Polynomial Reg.)')
plt.xlabel('Days starting from 30th Jan')
plt.ylabel('Total Confirmed Cases')
plt.show()

predictActiveCases(day)
predictRecoveredCases(day)
predictDeathCases(day)


