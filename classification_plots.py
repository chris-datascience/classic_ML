# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 19:27:09 2018

@author: Kris
"""

import numpy as np
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

old_settings = np.geterr()
np.seterr(divide='ignore')

def trapezoid_area(X1, X2, Y1, Y2):
    base = np.abs(X1 - X2)
    height_avg = (Y1 + Y2) / 2
    return base * height_avg

def ROC_create(labels, predicted_probabilities, figsize=(8,7), clf_name='clf', verbose=True, plot_ROC=True, plot_lift=True, plot_FDR=True, plot_recall_precision=True):
    """
    - Code ROC & AUC acc. to <<An introduction to ROC analysis>> by Tom Fawcett, 
      Pattern recognition Letters 27 (2006), pg. 861-874.
    - Code Lift acc. to <<Data Science for business>> by Provost & Fawcett (2013)
    """
    L = list(zip(list(labels), list(predicted_probabilities)))
    TP = FP = 0
    TP_prev = FP_prev = 0
    P = np.sum(labels)
    if P==0:
        print('\nNo positives in entire set!\n')
        return
    n_instances = len(labels)
    N = n_instances - P
    AUC = 0
    ROC = []; lift = []; TPR = []; Precision = []; FDR = [] # False Discovery Ratio
    prob_prev = -1
    L_sorted = sorted(L, key=lambda x: x[1])
    for i,(label, prob) in enumerate(L_sorted):
        if prob!=prob_prev:
            ROC.append((FP / N, TP / P))
            AUC += trapezoid_area(FP, FP_prev, TP, TP_prev)
            prob_prev = prob
            TP_prev = TP
            FP_prev = FP
        TP += (label==1)*1  # equivalent to if label==1: TP+=1
        FP += (label==0)*1  # ERROR! FPR AND TPR ARE WRONG WAY AROUND!
        # update lift
        perc_of_test_instances_so_far = (i + 1) / n_instances
        lift.append((perc_of_test_instances_so_far, (TP/P) / perc_of_test_instances_so_far))
        # update FDR
        try:
            FDR.append(FP / TP)
        except ZeroDivisionError:
            FDR.append(np.nan)
        # update Rec & Precc
        TPR.append(TP / P)
        try:
            Precision.append(TP / (TP + FP))
        except ZeroDivisionError:
            Precision.append(np.nan)
            print('got it')
            
    ROC.append((FP / N, TP / P))
    AUC += trapezoid_area(N, FP_prev, N, TP_prev)
    AUC /= (P * N)
    
    # Plotting ROC
    if plot_ROC:
        plt.figure(figsize=figsize)
        plt.plot([0,1], [0,1], 'k:', alpha=.6)
        plt.fill_between([a for a,_ in ROC], [b for _,b in ROC], color='b', alpha=.2)
        plt.plot([a for a,_ in ROC], [b for _,b in ROC], 'b-', lw=2, label=clf_name)
        plt.title('ROC', fontsize=15)
        plt.legend(fontsize=14, loc=4)
        plt.xlabel('FPR (-)', fontsize=14)
        plt.ylabel('TPR (-)', fontsize=14)
    # AUC to output
    if verbose:
        print('\nAUC = %.3f' % AUC)

    # Plotting Lift curve:
    if plot_lift:
        plt.figure(figsize=figsize)
        plt.plot([0,1], [1,1], 'k:', alpha=.6)
        plt.plot([a for a,_ in lift], [b for _,b in lift], 'c-', lw=2, label=clf_name)
        plt.legend(fontsize=14, loc=1)
        plt.title('Lift of classifier', fontsize=15)
        plt.xlabel('percentage of test instances (decreasing by score)', fontsize=14)
        plt.ylabel('lift (-)', fontsize=14)
        plt.ylim(0, np.ceil(max([b for _,b in lift])))

    if plot_recall_precision:
        plt.figure(figsize=figsize)
        plt.plot(TPR, Precision, 'g-', lw=2)
        plt.title('Recall-Precision', fontsize=15)
        plt.xlabel('TPR = Recall (-)', fontsize=14)
        plt.ylabel('Precision (-)', fontsize=14)
        plt.ylim(0, 1)
    if verbose:
        R = np.array(TPR)
        print('\nMaximum Precision is %.3f, reached at Recall of %.3f' % (max(Precision), R[np.nanargmax(Precision)]))

    if plot_FDR:
        plt.figure(figsize=figsize)
        plt.plot(TPR, FDR, 'm-', lw=2)
        plt.title('False Discovery Ratio', fontsize=15)
        plt.xlabel('TPR = Recall (-)', fontsize=14)
        plt.ylabel('FP : TP', fontsize=14)
        plt.ylim(0, 30)
    if verbose:
        Recall = np.array(TPR)
        try:
            print('\nBest False Discovery Rate: %i FPs for each TP, reached at Recall of %.3f.' % (min(FDR), Recall[np.nanargmin(FDR)]))
        except:
            print('\nCouldnt compute best FDR.')

def ROC_vertical_averaging():
    """
    Code acc. to <<An introduction to ROC analysis>> by Tom Fawcett, 
      Pattern recognition Letters 27 (2006), pg. 861-874.
    """
    pass

def ROC_horizontal_averaging():
    """
    Code acc. to <<An introduction to ROC analysis>> by Tom Fawcett, 
      Pattern recognition Letters 27 (2006), pg. 861-874.
    """
    pass


if __name__=='__main__':
#    N = 200
#    labels_random = np.random.randint(low=0, high=2, size=N)
#    predicted_probabilities_random = np.random.rand(N,).round(2)
#    ROC_create(labels_random, predicted_probabilities_random, clf_name='random clf')

# =============================================================================
#     LOAD EXAMPLE TEST RESULTS
# =============================================================================
    T = pd.read_csv('NB_testlabels.csv', nrows=10000) 
    labels_test = T['labels'].values
    predicted_probabilities = T['pred_prob'].values
#    ROC_create(1 - labels_test, predicted_probabilities)
    
# =============================================================================
#    REFERENCE STATS from SKLEARN
# =============================================================================
    print(roc_auc_score(labels_test, predicted_probabilities))
    
    fpr, tpr, thresholds = roc_curve(labels_test, predicted_probabilities, pos_label=1)
    plt.figure(); plt.plot(fpr, tpr, 'b-'); plt.title('ROC', fontsize=15); plt.xlabel('FPR', fontsize=14); plt.ylabel('TPR = Recall', fontsize=14)
    
    Prec, Rec, thresholds  = precision_recall_curve(labels_test, predicted_probabilities, pos_label=1)
    plt.figure(); plt.plot(Rec, Prec, 'g-'); plt.xlabel('Recall', fontsize=14); plt.ylabel('Precision', fontsize=14)

    FalseDiscovRatio = (1 - Prec) / Prec
    plt.figure(); plt.plot(Rec, FalseDiscovRatio, 'm-'); plt.xlabel('Recall', fontsize=14); plt.ylabel('FP / TP', fontsize=14); plt.title('False Discovery Ratio', fontsize=15)
    
    

