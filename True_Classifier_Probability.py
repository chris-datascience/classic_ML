# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 15:25:46 2018

@author: Kris
"""

"""
Topic:
    Classification performance beyond ROC, AUC for ensemble methods such as Random Forest.
    Applying Bayes Rule for deriving range of probabilities.
Source:
    https://medium.com/bcggamma/deriving-true-probability-from-an-ensemble-classifier-4417f7a67ac4
"""

import numpy as np
def plotcorrected(clf_score, true_labels):
  # true_labels: length n array of ground truth classes, boolean
  # clf_score: length n array of classifier scores
  
  nbins = 15
  bins = np.linspace(min(clf_score), max(clf_score), nbins, endpoint=True)
  # p(score|1)
  Pscoregiven1, _ = np.histogram(clf_score[true_labels], bins, density=True)
  # p(score|0) 
  Pscoregiven0, _ =   np.histogram(clf_score[np.logical_not(true_labels)], bins, density=True) 
  
  p1 = sum(true_labels)/true_labels.shape[0] #P(1)
  p0 = 1-p1 #P(0)
  
  # p(score|1)P(1)
  up = p1*Pscoregiven1 
  # p(score|1)P(1)+p(score|0)P(0)=P(score) 
  down = p1*Pscoregiven1 + p0*Pscoregiven0
  # desired P(1|score)
  true_probs = up/down
 
  return bins, true_probs