# -*- coding: utf-8 -*-
"""
Created on Thu May 17 17:30:34 2018

@author: erdbrca
"""

from __future__ import division
#imports and definitions
import numpy as np
import scipy.stats as stats
import scipy.spatial.distance as distance
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet as MCD
from sklearn.covariance import MinCovDet  # EmpiricalCovariance
"""
ad MCD:
Robust sampled covariance estimate called Minimum Covariance Determinant
See http://scikit-learn.org/stable/modules/covariance.html
http://scikit-learn.org/stable/auto_examples/covariance/plot_robust_vs_empirical_covariance.html#sphx-glr-auto-examples-covariance-plot-robust-vs-empirical-covariance-py
"""
  

"""
source: 
    http://nullege.com/codes/show/src%40p%40y%40Python-Numerics-HEAD%40MachineLearningScikitLearn%40outlier.py/23/scipy.spatial.distance.mahalanobis/python

See also:
    https://en.wikipedia.org/wiki/Mahalanobis_distance

Slightly adapted; among others to work in Python 3
"""

  
class Outlier_detection(object):
    def __init__(self, support_fraction = 0.95, verbose = True, chi2_percentile = 0.995):
        self.verbose = verbose
        self.support_fraction = support_fraction
        self.chi2 = stats.chi2
        self.mcd = MCD(store_precision = True, support_fraction = support_fraction)
        self.chi2_percentile = chi2_percentile
  
    def fit(self, X):
        """Prints some summary stats (if verbose is one) and returns the indices of what it consider to be extreme"""
        self.mcd.fit(X)
        mahalanobis = lambda p: distance.mahalanobis(p, self.mcd.location_, self.mcd.precision_  )
        d = np.array(list(map(mahalanobis, X))) #Mahalanobis distance values
        self.d2 = d ** 2 #MD squared  # <--- l2 norm only option?!
        n, self.degrees_of_freedom_ = X.shape
        self.iextreme_values = (self.d2 > self.chi2.ppf(0.995, self.degrees_of_freedom_) )  # boolean array showing outliers
        self.outlier_inds = np.nonzero(od.iextreme_values)[0]  # 
        if self.verbose:
            print("%.3f proportion of outliers at %.3f%% chi2 percentile, "%(self.iextreme_values.sum()/float(n), self.chi2_percentile))
            print("with support fraction %.2f."%self.support_fraction)
        return self
  
    def plot(self, log=False, sort=False ):
        """
        log: transform the distance-sq to a log
        sort: sort the data according to distance before plotting
        """
        n = self.d2.shape[0]
        fig = plt.figure()
  
        x = np.arange(n)
        ax = fig.add_subplot(111)
  
        transform = (lambda x: x ) if not log else (lambda x: np.log(x))
        chi_line = self.chi2.ppf(self.chi2_percentile, self.degrees_of_freedom_)     
        chi_line = transform( chi_line )
        d2 = transform( self.d2 )
        if sort:
            isort = np.argsort( d2 )    
            ax.scatter(x, d2[isort], alpha = 0.7, facecolors='none' )
            plt.plot( x, transform(self.chi2.ppf( np.linspace(0,1,n),self.degrees_of_freedom_ )), c="r", label="distribution assuming normal" )
        else:
            ax.scatter(x, d2 )
            extreme_values = d2[ self.iextreme_values ]
            ax.scatter( x[self.iextreme_values], extreme_values, color="r" )
  
        ax.hlines( chi_line, 0, n, 
                        label ="%.1f%% $\chi^2$ quantile"%(100*self.chi2_percentile), linestyles = "dotted" )
        ax.legend()
        ax.set_ylabel("distance squared")
        ax.set_xlabel("observation")
        ax.set_xlim(0, self.d2.shape[0])
        plt.show()
        
#        if plot_2d:
#            if self.degrees_of_freedom_!=2:
#                print('Dataset dimensions do not allow 2D plot.')
#            else:

# =============================================================================
#     # TO DO:   ADD 3D VERSION (SHOULD HAVE THIS SOMEWHERE)
# =============================================================================

# =============================================================================
#     # TO DO:   ADD ROBUST DEMO FROM SKLEARN (AND ADAPT), SEE BELOW:
# =============================================================================
    """ Robust Mahalanobis distance
        Sources:
            https://en.wikipedia.org/wiki/Mahalanobis_distance
            http://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html#sphx-glr-auto-examples-covariance-plot-mahalanobis-distances-py
            ^^ latter uses robust estimates of mu and covariance!
    """
    # fit a Minimum Covariance Determinant (MCD) robust estimator to data
    Mahal = pd.DataFrame(data=np.random(50,2)) # add
    X = Mahal.values  #MID_overview[['recentSales', 'CB_perc', 'HRW_perc']].values
    robust_cov = MinCovDet().fit(X)  # NB. robuster than standard covariance estimator EmpiricalCovariance().fit(X)
    Mahal['mahal_dist'] = robust_cov.mahalanobis(X - robust_cov.location_) #** (0.33)
    Mahal['rank_mahal'] = Mahal.mahal_dist.rank(ascending=True).astype(int)
    # Check mahal manually:
    #x = X - robust_cov.location_
    #qq = np.dot(x.T, np.linalg.inv(robust_cov.covariance_)).dot(x)  # problem: cannot invert singular matrix
    
    # Inspect most common (inliers) and UNcommon samples:
    #print(MID_overview[['recentSales', 'recentCB', 'rank_mahal']].sort_values(by=['rank_mahal'])[-10:])
    # CONCLUSION: CAN USE THIS FOR RANKING BUT NOT FOR OPTIMISATION (STRAIGHTFORWARD MINIMISATION); 
    # WE'VE MEASURED HOW COMMON/UNCOMMON THE PERFORMANCE OF EACH MID IS!
    
    
# =============================================================================
#     # TO DO: COMPARE TO RANSAC & ISOLATION FORES OR SOME OTHER METHODS
# =============================================================================

if __name__=='__main__':        
    # --MAIN--
    od = Outlier_detection()
#    X = np.vstack((np.arange(500), 1.5*np.arange(500) + 4*np.random.randn(500,2))  # dataset
    X = np.random.randn(500,2)
    od.fit(X)
    od.plot(log=False)
    
    plt.figure(figsize=(8,8))
    plt.plot(X[:,0], X[:,1], 'k.', alpha=.6)
    plt.plot(X[od.outlier_inds,0], X[od.outlier_inds,1], 'co', markersize=10, alpha=.5, label='outlier')
    plt.legend()
