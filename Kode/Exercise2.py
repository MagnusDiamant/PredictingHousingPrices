import numpy as np
import linreg 
import matplotlib.pyplot as plt 
import math 
import pandas 
from sklearn.model_selection import LeaveOneOut

z_data = pandas.read_csv("men-olympics-100.txt", header = None, sep=" ")

mYear = z_data.iloc[:, 0]
mRun = z_data.iloc[:, 1]

lambdaGen = np.logspace(-8, 0, 100, base = 10)

lambdaErr = np.zeros(np.size(lambdaGen))

for i in range (np.size(lambdaGen)):
    LOO = LeaveOneOut()
    ErrSum = 0 
    
    for x,y in LOO.split(mYear):
        train_t, test_t = mRun[x], mRun[y]
        train_x, test_x = mYear[x], mYear[y]
        ones = np.ones((np.size(train_x), 1))
        
        train_x = np.matrix(train_x).transpose()
        train_x = np.concatenate((ones, train_x), axis = 1)
        numbers = np.size(mYear) 
        
        w = np.dot(np.linalg.inv((np.dot(train_x.transpose(),
            train_x) + numbers * lambdaGen[i] * np.identity(2))),
            (np.dot(train_x.transpose(), train_t)).transpose())
        
        ErrSum += sum((test_t - (w[0,0] + w[1,0] * test_x))**2)
    lambdaErr[i] = math.sqrt(ErrSum)

bestLambda = min(lambdaErr)
    
print("Opgave 2a")
plt.xlabel("Lambda")
plt.ylabel("Errors")
plt.plot(np.log10(lambdaGen), lambdaErr)
plt.show()

print("Best value of Lambda:")
print(bestLambda)

print("Regression coefficients for best possible lambda:")
print(np.dot(np.linalg.inv((np.dot(train_x.transpose(), train_x) + numbers * 
                                  bestLambda * np.identity(2))), (np.dot(train_x.transpose(), train_t)).transpose()))

print("Regression coefficients for lambda = 0:")
print(np.dot(np.linalg.inv((np.dot(train_x.transpose(), train_x) + numbers * 
                                  0 * np.identity(2))), (np.dot(train_x.transpose(), train_t)).transpose()))

lambdaGen = np.logspace(-8, 0, 100, base = 10)

lambdaErr = np.zeros(np.size(lambdaGen))

for i in range (np.size(lambdaGen)):
    LOO = LeaveOneOut()
    ErrSum = 0 
    
    for x,y in LOO.split(mYear):
        train_t, test_t = mRun[x], mRun[y]
        train_x, test_x = mYear[x], mYear[y]
        ones = np.ones((np.size(train_x), 1))
        
        train_x = np.matrix(train_x).transpose()
        train_x = np.concatenate((ones, train_x, np.power(train_x, 2), 
                  np.power(train_x, 3), np.power(train_x, 4)), axis = 1)
        numbers = np.size(mYear) 
        
        w = np.dot(np.linalg.inv((np.dot(train_x.transpose(), train_x)
            + numbers * lambdaGen[i] * np.identity(5))), 
            (np.dot(train_x.transpose(), train_t)).transpose())
        
        ErrSum += sum((test_t - (w[0,0] + w[1,0] * test_x + w[2,0]* 
                  test_x**2 + w[3,0] * test_x**3 + w[4,0]*test_x**4))**2)
    
    lambdaErr[i] = math.sqrt(ErrSum)

bestLambda = min(lambdaErr)

print("Opgave 2b")
plt.xlabel("Lambda")
plt.ylabel("Errors")
plt.plot(np.log10(lambdaGen), lambdaErr)
plt.show()

print("Best value of Lambda:")
print(bestLambda)

print("Regression coefficients for best possible lambda:")
print(np.dot(np.linalg.inv((np.dot(train_x.transpose(), train_x) + numbers * 
                                  bestLambda * np.identity(5))), (np.dot(train_x.transpose(), train_t)).transpose()))

print("Regression coefficients for lambda = 0:")
print(np.dot(np.linalg.inv((np.dot(train_x.transpose(), train_x) + numbers * 
                                  0 * np.identity(5))), (np.dot(train_x.transpose(), train_t)).transpose()))