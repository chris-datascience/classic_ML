# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:30:07 2016

@author: christiaan.erdbrink

MLE for 1D-regression with prescribed structure.
"""

import numpy as np
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

def feval_lin(t, C):
    return C[0]*t + C[1]

def feval_para(t, C):
    return C[0]*t**2 + C[1]*t + C[2]

def feval_exp(t,C):
    return C[0]*np.exp(C[1]*t) + C[2]
    
def ml_constants(y_model, y_target):
    n = float(len(y_target))   
    beta1 = (n * np.sum(y_model * y_target) - np.sum(y_model) * np.sum(y_target)) / (n * np.sum(y_model ** 2) + (np.sum(y_model))**2)
    beta0 = (1/n) * np.sum(y_target - beta1 * y_model) 
    sigma2 = (1/n) * np.sum((y_target - beta0 - beta1 * y_model)**2) 
    return (beta0, beta1, sigma2)

def loglikelihood(C, x, y, structure):
    if structure=='exponential':
        b0, b1, S = ml_constants(feval_exp(x, C), y)
    elif structure=='linear':
        b0, b1, S = ml_constants(feval_lin(x, C), y)
    elif structure=='parabola':
        b0, b1, S = ml_constants(feval_para(x, C), y)
    return -1*(float(len(y))/2) * (np.log(2*np.pi*S) + 1)*np.ones((3))

def mle(x, y, x_eval_pts, form='linear'):
    if form=='linear':
        lr = LR()
        lr.fit(np.expand_dims(target_x, axis=1), np.expand_dims(target_y_noisy, axis=1))
        params0 = np.hstack((lr.coef_[0], lr.intercept_))
        params = leastsq(loglikelihood, params0, args=(x, y, form))
        tx = feval_lin(x, params[0])    
    elif form=='parabola':
        x_sq = x.copy()**2
        x_train = np.vstack((x, x_sq)).T
        lr = LR()
        print(x_train.shape)
        lr.fit(x_train, np.expand_dims(target_y_noisy, axis=1))
        params0 = np.hstack((lr.coef_[0][0], lr.coef_[0][1], lr.intercept_))        
        params = leastsq(loglikelihood, params0, args=(x, y, form))
        tx = feval_para(x, params[0])
    elif form=='exponential':
        params0 = np.array([10,0.1,10]) #np.random.randn(3)
        params = leastsq(loglikelihood, params0, args=(x, y, form))
        tx = feval_exp(x, params[0])    
    beta0,beta1,sigma2 = ml_constants(tx, y)
    y_model = beta0 + beta1*x   
    std_high = y_model + np.sqrt(sigma2)
    std_low = y_model - np.sqrt(sigma2)    
    return y_model, std_high, std_low


# ----MAIN----
if __name__=='__main__':
    plt.close('all')
    
    # Create target dataset:
    target_x = np.linspace(40,130,num=20)
    target_y_original = 20*np.exp(0.05*target_x) + 100
    target_y_noisy = target_y_original + 500.*np.random.randn(len(target_x))
    x_eval_points = np.linspace(50, 120, num=50)  # where to evaluate
    
    structure = 'parabola'   # <---------- SELECT
    slice_location = 80    # <---------- SELECT OUTPUT LOCATION
    
    # Use MLE to find standard deviation bars of noise for best fitting model:
    (model_y_mle, s1, s2) = mle(target_x, target_y_noisy, x_eval_points, form=structure)
    
    # Plotting result
    plt.figure(figsize=(8,6))
    plt.plot(target_x, target_y_noisy ,'k.', label='data') 
    plt.plot(x_eval_points, model_y_mle, 'r-', label='model')
    plt.plot(x_eval_points, s1, 'b--',label='std')
    plt.plot(x_eval_points, s2, 'b--')
    plt.legend(loc='best')
    plt.title('Fitted model: %s' % structure, fontsize=15)
    
    
    
    