import numpy as np 

# NOTE: This template makes use of Python classes. If 
# you are not yet familiar with this concept, you can 
# find a short introduction here: 
# http://introtopython.org/classes.html

class LinearWeightRegression():


    def __init__(self):
        
        pass
            
    def fit(self, X, t):
      
        X = np.array(X).reshape((len(X), -1))
        t = np.array(t).reshape((len(t), 1))

        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        A = np.diagflat(t, k=0)
        A = np.square(A)
        
        
        xt = X.T
        AX = A.dot(X)
        a = xt.dot(AX) 
        
        At = A.dot(t)
        b = xt.dot(At)
        
        self.w = np.linalg.solve(a,b)
        

    def predict(self, X):
                   

        X = np.array(X).reshape((len(X), -1))

        # prepend a column of ones
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)

        # compute predictions
        predictions = np.dot(X, self.w)
        
        return predictions 
