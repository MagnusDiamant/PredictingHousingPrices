import numpy as np
import matplotlib as mpl 
# load data
train_data = np.loadtxt("boston_train.csv", delimiter=",")
test_data = np.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (a) compute mean of prices on training set
print("Opgave a")
resultA = np.mean(t_train)
print("Result: %g" % resultA)

# (b) RMSE function

def rmse(t, tp):
    print("Opgave b")
    RMSE = np.sqrt(np.mean((t - tp) ** 2))
    print("Result: %g" % RMSE)
    return RMSE

rmse(resultA, t_test)
#     ...

# (c) visualization of results
print("Opgave c")
t_plot = t_train.reshape((len(t_train), 1))
t_mean = np.ones(len(t_test),)*resultA

mpl.pyplot.plot(t_test, t_mean, '.', color='black');

