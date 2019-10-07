# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:55:26 2019

@author: erdbrca
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm, ttest_ind

colors = 'grbymc'*8


def sample_variance(obs):
    """
        Note: applying (n-1) in denominator, not n as in np.sqrt
    """
    return np.sqrt(np.sum([(x - np.mean(obs))**2 for x in obs]) / (len(obs)-1))
    
def plot_student(nu_values, x_range=[-8,8], title=""):
    x = np.linspace(-8,8,101)
    plt.figure(figsize=(9,7))
    for i,nu in enumerate(nu_values):
        #plt.plot(x, t.pdf(x, df=1, loc=1, scale=1), 'g-', label='nu=1')
        #plt.plot(x, t.pdf(x, df=5, loc=1, scale=1), 'r-', label='nu=5')
        plt.plot(x, t.pdf(x, df=nu, loc=1, scale=1), colors[i]+'-', label='nu=%i'%(int(nu)))  # 'df' is parameter nu
    plt.plot(x, norm.pdf(x, loc=1, scale=1), 'k-', label='normal')
    plt.legend(fontsize=15)
    plt.title(title, fontsize=16)
    
def classic_student():
    pass

def independent_student(X1, X2):
    """
        aka. Welch's t-test for independent two-sample test of difference sample size (but equal variance!)
        https://en.wikipedia.org/wiki/Welch%27s_t-test
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    """
    #p_Welch = test_ind()
    pass

"""
Sources:
    Efron & Hastie textbook
    https://en.wikipedia.org/wiki/Student%27s_t-test#Independent_.28unpaired.29_samples
    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
"""

def demo_Student_t_pdf():
    x = np.linspace(-6,6,1000)
    plt.figure()
    colors = list('brkmcyg')*3
    for i,dof in enumerate([1,2,5,10,20,50,100]):
        plt.plot(x, t.pdf(x, dof), colors[i]+'-', lw=1, alpha=0.7, label='dof = '+str(dof))
    plt.legend()
    plt.title("Student's t distribution",fontsize=15)

def two_sample_Student_t_test(data1, data2):
    """
    --Independent two-sample test--
    Assuming Gaussian distributions and equal variances, but unequal sample sizes are allowed. 
    Hypothesis H0: mu_1 == mu__2
    """
    sample_mean1 = np.mean(data1)
    sample_mean2 = np.mean(data2)
    n1 = len(data1)
    n2 = len(data2)
    degrees_of_freedom = n1 + n2 - 2
    s_p = np.sqrt(((n1-1)*(np.std(data1)**2) + (n2-1)*(np.std(data2)**2)) / degrees_of_freedom)

    print('n1 = %i, n2 = %i' % (n1,n2))
    print('dof = %2.2f' % degrees_of_freedom)
    print('s_p = %1.4f' % s_p)
    
    t_statistic = (sample_mean1 - sample_mean2) / (s_p * np.sqrt(1./n1 + 1./n2))
    p_value = (1 - t.cdf(abs(t_statistic), degrees_of_freedom))*2  # Look up from Student's t-distribution
    confidence_interval = 1. - p_value
    return t_statistic, p_value, confidence_interval
    
    
def two_sample_Welch_t_test(data1, data2, scale_estimator=lambda x: np.std(x)):
    """
    --Independent two-sample test--
    Assuming Gaussian distributions and UNequal variances and unequal sample sizes. 
    Hypothesis H0: mu_1 == mu__2
    scale_estimator is a function that estimates the square root of the variance (~st.dev.)
    """
    sample_mean1 = np.mean(data1)
    sample_mean2 = np.mean(data2)
    n1 = len(data1)
    n2 = len(data2)
    s1 = scale_estimator(data1)
    s2 = scale_estimator(data2)
    s_delta = np.sqrt((s1**2)/n1 + (s2**2)/n2)
    t_statistic = (sample_mean1 - sample_mean2) / s_delta
    degrees_of_freedom = s_delta**4 / ((s1**2/n1)**2/(n1 - 1) + (s2**2/n2)**2/(n2 - 1))  # Welch–Satterthwaite equation
    p_value = (1 - t.cdf(abs(t_statistic), degrees_of_freedom))*2  # Look up from Student's t-distribution
    return p_value, t_statistic, degrees_of_freedom


def two_sample_Kolmogorov_Smirnov_test(data1, data2):
    """    
    A nonparametric test of the equality of two 1D distribitions.
    The K-S statistic quantifies a distance between the empirical distribution functions of two samples. 
    The null distribution of this statistic is calculated under the null hypothesis that the samples are drawn from the same distribution. 
    """
    
def BEST():
    """
        Bayesian t-test:
            https://docs.pymc.io/notebooks/BEST.html
            <article by Kruschke>
            <Bayesian methods for hackers>
    """
    pass

    
if __name__=='__main__':
    #plot_student([1,5,8], title='DEMO')
    
    # Dummy test sets to play around with
    #A1 = 12. + 1.*np.random.randn(20,)
    #A2 = 100. + .3*np.random.randn(4,)
    
    # Validation example 3: https://en.wikipedia.org/wiki/Welch%27s_t-test
    A1 = np.array([19.8, 20.4, 19.6, 17.8, 18.5, 18.9, 18.3, 18.9, 19.5, 22.])
    A2 = np.array([28.2, 26.6, 20.1, 23.3, 25.2, 22.1, 17.7, 27.6, 20.6, 13.7, 23.2, 17.5, 20.6, 18., 23.9, 21.6, 24.3, 20.4, 24., 13.2])
    
    P, t_stat, dof = two_sample_Welch_t_test(A1, A2, scale_estimator=sample_variance)
    print('\nWelch:\n p-value = %2.3f (check: 0.036)\n t-statistic = %2.3f (check:-2.22)\n degrees of freedom = %2.1f (check: 24.5)' % (P, t_stat, dof))
    
    B1 = 12. + np.random.randn(25,)
    B2 = 12.5 + np.random.randn(15,)
    t_statistic, p_value, _ = two_sample_Student_t_test(B1, B2)
    print('\nClassic two-sample t-test: %2.3f, %2.4f' % (t_statistic, p_value))
    