import numpy
import pandas
import linreg
import matplotlib.pyplot as plt

# load data
train_data = numpy.loadtxt("boston_train.csv", delimiter=",")
test_data = numpy.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (b) fit linear regression using only the first feature
print("Opgave b")
model_single = linreg.LinearRegression()
model_single.fit(X_train[:,0], t_train)
print(model_single.w)
print("Ud fra disse to værdier w0 og w1, kan vi se, at hvis CRIM er 0, vil husets værdi (t) være 23,6351 = w0. w1 er negativ, hvilket fortæller os, at jo højere crime rate jo mere falder husværdien (t).")

# (c) fit linear regression model using all features
print("Opgave c")
model_all = linreg.LinearRegression()
model_all.fit(X_train, t_train)
print(model_all.w)

# (d) evaluation of results
print("Opgave d")
predict_single = model_single.predict(X_test[:,0])
plt.scatter(predict_single, t_test)
plt.show()
predict_all = model_all.predict(X_test)
plt.scatter(predict_all, t_test)
plt.show()

def rmse(t, tp):
    RMSE = np.sqrt(np.mean((t - tp) ** 2))
    return RMSE

print("Predict_single RMSE: %g" % rmse(predict_single, t_test))
print("Predict_single RMSE: %g" % rmse(predict_all, t_test))

