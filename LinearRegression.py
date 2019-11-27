#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:43:13 2019

@author: aims
"""

class LinearRegression(object):
    
    def __init__(self,lr= 0.01, tolerance = 0.001, iterations = 20):
        self.lr = lr
        self.tolerance = tolerance
        self.iterations = iterations


    
    def compute_gradient(self, y, theta, X):
        
        return (1/(X.shape[0])) *np.sum((X@theta -y)*X, axis=0) #gradient
    
    
    
    def cost(self, y, theta, X):
        
        return (1/(X.shape[0])) * np.sum((y - X@theta)**2)
    
    
    
    def fit(self,X, y):
        X_copy = X.copy()

        X_copy = np.hstack((np.ones((X.shape[0],1)), X_copy))
        
        self.theta = np.zeros((X_copy.shape[1],1))
        
        for i in range(self.iterations):
            
            gradient = self.compute_gradient(y, self.theta, X_copy)
            self.theta = self.theta - self.lr*gradient.reshape(-1,1)
            print(self.cost(y,self.theta, X_copy))
            
            
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        prediction = X @self.theta
        return prediction