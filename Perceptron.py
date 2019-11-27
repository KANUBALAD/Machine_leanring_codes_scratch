# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

def perceptron_weights(X):
    lr=0.05
    eps=1e-10
    weights_w = np.zeros((X.shape[1]))
    signs_y= [np.sign(weights_w.dot(X[i,:])) for i in range(X.shape[0])]
    last_column = X.iloc[:,-1]
    misclassified_idx = [i for i in range(X.shape[0]) if last_column[i] != signs_y[i]]
    if misclassified_idx==[]:
        return weights_w
    else:
        wprev= weights_w 
        features=X.drop(last_column,axis =1)
        gradient = -sum([last_column[i]*features[i] for i in misclassified_idx])
        weights_w  -= lr*gradient                    
        while(np.linalg.norm(wprev-weights_w)>eps):
            signs_y= [np.sign(weights_w.dot(X[i,:])) for i in range(X.shape[0])]
            misclassified_idx = [i for i in range(X.shape[0]) if last_column[i] != signs_y[i]]
            wprev = weights_w 
            features=X.drop(last_column,axis =1)
            gradient = -sum([last_column[i]*features[i] for i in misclassified_idx])
            weights_w  -= lr*gradient
        return weights_w