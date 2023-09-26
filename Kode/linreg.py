import numpy as np 

# NOTE: This template makes use of Python classes. If 
# you are not yet familiar with this concept, you can 
# find a short introduction here: 
# http://introtopython.org/classes.html

class LinearRegression():
    """
    Linear regression implementation.
    """

    def __init__(self):
        
        pass
            
    def fit(self, X, t):
        """
        Fits the linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of shape [n_samples, 1]
        """        
        X = np.array(X).reshape((len(X), -1))
        t = np.array(t).reshape((len(t), 1))

        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        
        xt = X.T
        a = xt.dot(X) 
        
        b = xt.dot(t)
        
        self.w = np.linalg.solve(a,b)
        

    def predict(self, X):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of shape [n_samples, 1]
        """                     

        X = np.array(X).reshape((len(X), -1))

        # prepend a column of ones
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)

        # compute predictions
        predictions = np.dot(X, self.w)
        
        return predictions 
    
    def least_squares(self, X, t, z):
        X = np.array(X).reshape((len(X)), -1)
        t = np.array(t).reshape((len(t)), -1)
        
        z = len(X) * z *np.identity(len(X))
        z = np.add(XTX, z)
        
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X-ones*1896), axis=1)
        
        XTX = X.T.dot(X) 
        XTt = X.T.dot(t)
        
        a = np.linalg.solve(z, XTt)
        
        
        
        
        