# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 21:27:54 2020

@author: Kris
"""

import numpy as np
from scipy.stats import poisson, nbinom
from scipy.optimize import minimize
from scipy.special import gamma
import matplotlib.pyplot as plt

# =============================================================================
# VANILLA POISSON 
# =============================================================================
def pre_train_fact(y_max=51):
    factorials = {}#np.empty(y_max, dtype=np.ulonglong)
    for y in range(y_max):
        factorials[y] = float(np.math.factorial(y))
    return factorials

def poisson_pmf(mu, y):
    """ Not vectorised. """
    if y<0:
        print('WARNING! Poisson not defined for values below 0. Zero probability mass returned.')
        return 0        
    return (np.exp(-mu) * mu**y) / (np.math.factorial(y))

def poisson_loglikel(mu, observations):
    factorial_values = np.array([np.log(fact[i]) for i in obs], dtype=np.float)
    return -np.sum(observations * np.log(mu) - mu - factorial_values)
    
# Observations for training:
#obs = np.array([0,3,0,2,1,1,1,0,5,4,7,2,2,3,3,1,1,1,4,14], dtype=int)
obs = poisson(2.3).rvs(40)  # 40 random samples from Poisson with chosen parameter

# MLE Poisson
fact = pre_train_fact()
res = minimize(poisson_loglikel, 2., args=obs, method='Nelder-Mead')  #, bounds=[(0.01,20)]
print('\nOptimired model parameter Poisson model:', res.x)
poisson_fitted = poisson(res.x)
x = np.arange(0,np.max(obs)+1)
#plt.figure(figsize=(7,6))
fig, ax = plt.subplots(1, 1)
ax.vlines(x, 0, np.bincount(obs)/np.sum(np.bincount(obs)), colors='g', linestyles='-', lw=12, alpha=.2, label='observed')
ax.vlines(x, 0, poisson_fitted.pmf(x), colors='k', linestyles='-', lw=3, label='P model')
ax.legend(loc='best')

# =============================================================================
# ZERO-TRUNCATED POISSON  (ZTP)
# =============================================================================
def zero_trunc_poisson_pmf(mu, y):
    """ mu is model parameter, y is value.
        NB. not vectorised for y.
    """
    if y<1:
        print('WARNING! ZTP not defined for values below 1. Zero probability mass returned.')
        return 0
    return ( np.exp(-mu) * (mu**y) ) / ( (1 - np.exp(-mu)) * np.math.factorial(y) )

fig, ax = plt.subplots(1, 1)
x = np.arange(0,10)
ZTP_pmf = np.array([zero_trunc_poisson_pmf(2.3,i) for i in x])
ax.vlines(x, 0, ZTP_pmf, colors='b', linestyles='-', lw=3, label='ZTP model')
ax.legend(loc='best')
