#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:51:56 2019

@author: aims
"""

class LogisticRegression(object):
    
    def __init__(self,lr= 0.1, tolerance = 0.1, iterations = 10):
        self.lr = lr
        self.tolerance = tolerance
        self.iterations = iterations

    def sigmoid(self, X, theta):
        return 1/(1+np.exp(-X@self.theta))
    
    def compute_gradient(self, y, theta, X):
          
        return (1/X.shape[0])*np.sum((y-self.sigmoid(X, theta))*X, axis = 0)
       
    
    def cost(self, y, theta, X):
        return np.sum(-y*np.log(self.sigmoid(X, theta))- (1-y)*np.log(1-self.sigmoid(X, theta)))
        
   
    def fit(self,X, y):
        X_copy = X.copy()
        X_copy = np.hstack((np.ones((X_copy.shape[0],1)), X_copy))
        self.theta = np.zeros((X_copy.shape[1],1))
        
        
        for i in range(self.iterations):
            h = self.sigmoid(X_copy, self.theta)          
            gradient = self.compute_gradient(y,h, X_copy)
            self.theta = self.theta - self.lr*gradient.reshape(-1,1)
            print(self.cost(y,h, X_copy))
        
        
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        prediction = X @self.theta
        return prediction