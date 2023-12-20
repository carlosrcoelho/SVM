import numpy as np

# SVM class

class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):       
        self.lr = learning_rate           # learning rate
        self.lambda_param = lambda_param  # regularization parameter
        self.n_iters = n_iters           # number of iterations
        self.w = None                    # weights
        self.b = None                   # bias

    
    def fit(self, X, y):                 # training function
        n_samples, n_features = X.shape  # number of samples and features in the dataset
        y_ = np.where(y <= 0, -1, 1)     # if y <= 0, y_ = -1, else y_ = 1 it converts the labels to -1 and 1
        self.w = np.zeros(n_features)    # weights initialization
        self.b = 0                       # bias initialization 

        # gradient descent
        for _ in range(self.n_iters):    # loop over the number of iterations
            for idx, x_i in enumerate(X):   # loop over the samples that gives the index and the sample
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1    # condition for the gradient descent
                if condition:                                                # if the condition is true
                    self.w -= self.lr * (2 * self.lambda_param * self.w)     # update the weights
                else:                                                        # if the condition is false
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))    # update the weights
                    self.b -= self.lr * y_[idx]                                                    # update the bias


    def predict(self, X):                # prediction function
        linear_output = np.dot(X, self.w) - self.b    # y = wx - b is the linear function
        return np.sign(linear_output)                 # sign(y) is the prediction function