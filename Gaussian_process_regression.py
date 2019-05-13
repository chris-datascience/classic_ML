# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

"""
GOAL:
Interpolation/extrapolation with confidence intervals using Gaussian Process Regression a.k.a. Kriging.

REFS:
<<Gaussian processes for regression: a quick introduction>> by M. Ebden(2008)
https://en.wikipedia.org/wiki/Kriging
https://en.wikipedia.org/wiki/Regression-Kriging
"""

def kron_delta(X, Y):
    if X==Y:  return 1
    else:  return 0

def k_squared_exp(x, y, sigma_f, sigma_n, l):
    """Squared exponential covariance function.
    sigma_f is max. allowable covariance,
    l is length parameter,
    sigma_n is ..."""
    return (sigma_f**2) * np.exp( (-(x - y)**2.) / (2.*l**2) ) + (sigma_n**2) * kron_delta(x,y)

def covariance_matrix_K(x_vector, sigma_f, sigma_n, l):  # sigma_f, l are scalars; sigma_n is vector of same size as x_vector
    K = np.empty((len(x_vector), len(x_vector)), dtype=float)
    for i,xi in enumerate(x_vector):
        for j,xj in enumerate(x_vector):
            K[i,j] = k_squared_exp(xi, xj, sigma_f, sigma_n[i], l)
    return K

def K_new_scalar(x_vector, x_new, sigma_f, sigma_n, l):  # x_new is scalar
    K_ = np.empty_like(x_vector)
    for i,xi in enumerate(x_vector):
        K_[i] = k_squared_exp(xi, x_new, sigma_f, sigma_n[i], l)
    return K_
    
def K_new_vector(x_vector, x_new, sigma_f, sigma_n, l): # x_new is vector with desired interpolation locations
    x_new = x_new.reshape(-1,)
    K_ = np.empty((len(x_new), len(x_vector)), dtype=float)
    for j,xn in enumerate(x_new):
        K_temp = np.empty_like(x_vector)
        for i,xv in enumerate(x_vector):
            K_temp[i] = k_squared_exp(xv, xn, sigma_f, sigma_n[i], l)
        K_[j,:] = K_temp.reshape(1,-1)
    return K_

def GP_regression(K, K_, K__, ydata):  # Gaussian Process Regression routine
    y_mean = np.dot(np.dot(K_, np.linalg.inv(K)), ydata)
    y_var = np.diag( K__ - np.dot(np.dot(K_, np.linalg.inv(K)), K_.T) )
    confidence_bar_low = y_mean - 1.96*np.sqrt(y_var)
    confidence_bar_high = y_mean + 1.96*np.sqrt(y_var)  # 95% confidnece bounds
    return y_mean, y_var, confidence_bar_low, confidence_bar_high

def log_likelihood(w, xdata, ydata, sigma_n):
    """log-likelihood function to maximize (MLE) to estimate good values for sigma_f and l.
    Nota bene: ML cost function is multiplied by -1 such that minimization algorithms can be used.
    K is the covariance matrix of xdata"""
    sigma_f, l = w  # unpacking weights vector
    K = covariance_matrix_K(xdata, sigma_f, sigma_n, l)
    return -( -.5 * np.dot(np.dot(ydata.reshape(1,-1), np.linalg.inv(K)), ydata) -.5*np.log(np.linalg.det(K)) - (len(xdata)/2.)*np.log(2*np.pi) )

def MLE_model_parameters(xdata, ydata, sigma_n):
    # TO DO
    w0 = np.array([.5, .5])
    w = optimize.minimize(log_likelihood, w0, args=(xdata, ydata, sigma_n), method='Nelder-Mead')
    sigma_f, l = w.x
    return sigma_f, l

def GPR_engine(xdata, ydata, sigma_n, x_new):  
    """ This is the all-in-one application, to be called from the script externally.
    Regression at several locations at once.
    e.g. x_new = np.linspace(-2,.5,100)
    
    N.B. 
    Sigma_n should basically a measure of prediction interval.
    See https://en.wikipedia.org/wiki/Prediction_interval
    It's important that sigma_n doesn't contain zeros! """
    
    sigma_f, l = MLE_model_parameters(xdata, ydata, sigma_n)
    if np.isnan(sigma_f) or np.isnan(l):   sigma_f, l = 1., 1.
    print('\nModel parameters (MLE):  sigma_f=%2.3f, l=%2.3f\n' % (sigma_f, l))
    if 0. in sigma_n:
        print('\nZeros found in sigma_n!\n')
        sigma_n[sigma_n==0.] = 1.25*np.mean(sigma_n[sigma_n!=0.])
    K = covariance_matrix_K(xdata, sigma_f, sigma_n, l)
    K_ = K_new_vector(xdata, x_new, sigma_f, sigma_n, l)            
    sigma_n_new = np.interp(x_new, xdata, sigma_n)
    K__ = covariance_matrix_K(x_new, sigma_f, sigma_n_new, l)
    y_mean, y_var, conf_low, conf_high = GP_regression(K, K_, K__, ydata)
    return y_mean, y_var, conf_low, conf_high
    
    
if __name__=='__main__':
    xdata = np.array([-1.5, -1., -.75, -.4, -.25, 0.])
    ydata = np.array([-1.6, -1.1, -.3, .2, .5, .9])
    #sigma_n = .3*np.ones((6,))
    #sigma_f = 1.27
    #l = 1. 
    sigma_n = np.array([.1, .25, .3, .35, .25, .25])
    sigma_f, l = MLE_model_parameters(xdata, ydata, sigma_n)
    print('\nModel parameters (MLE):  sigma_f=%2.3f, l=%2.3f\n' % (sigma_f, l))
    K = covariance_matrix_K(xdata, sigma_f, sigma_n, l)

#    # --regression at a single new x-location--
#    x_new = .2
#    K_ = K_new_scalar(xdata, x_new, sigma_f, sigma_n, l) 
#    K__ =  k_squared_exp(x_new, x_new, sigma_f, sigma_n, l)        
        
    # --Regression at several locations at once--
    x_new = np.linspace(-2,.5,100)
    K_ = K_new_vector(xdata, x_new, sigma_f, sigma_n, l)            
    sigma_n_new = np.interp(x_new, xdata, sigma_n)
    K__ = covariance_matrix_K(x_new, sigma_f, sigma_n_new, l)

    y_mean, y_var, conf_low, conf_high = GP_regression(K, K_, K__, ydata)

    plt.close('all')
    plt.figure(figsize=(8,6))
    plt.plot(x_new, y_mean, 'r-', lw=2)
    #plt.plot([x_new, x_new], [conf_low, conf_high], 'c-', lw=3, alpha=.3)
    plt.fill_between(x_new, conf_high, conf_low, where=conf_high>conf_low, facecolor='c', alpha=.25)
    plt.plot([xdata, xdata], [ydata-sigma_n, ydata+sigma_n], 'g-s', lw=3, alpha=.8)
    plt.plot(xdata, ydata, 'ko', markersize=11)
    plt.xlabel('x', fontsize=14)
    plt.xlabel('y', fontsize=14)
    plt.title('Gaussian Process regression demo', fontsize=15)