import numpy as np
import matplotlib.pyplot as plt

def confusion_metrics(TP,TN,FN,FP):
    if TP + FN > 0:
        recall = (float(TP)/float(TP+FN))
    else:
        recall = np.nan
    if TP + FP > 0:
        precision = (float(TP)/float(TP+FP))
    else:
        precision = np.nan
    if all([[~np.isnan(x) for x in [precision, recall]], recall + precision > 0]):
        F1_value = (2*recall*precision) / (recall + precision) 
    else:
        F1_value = 0
    if TN + FP > 0:
        TNR = TN / (TN + FP)
    else:
        TNR = 0
    if TP > 0:
        FDR = FP / TP
    else:
        FDR = np.nan
    return recall, precision, F1_value, TNR, FDR
    
def binary_classifier_administration(predicted_labels, test_labels, verbose=False):
    """
    For binary classification only.
    inputs:
        predicted labels, test labels, verbose, resp.
    outputs:
        TPR, precision, F, TNR, FDR
    
    TO DO:
        - Create separate routines for each metric, add False Discovery Ratio (=FP/TP), etc.
        - Improve the division by zero exceptions for each metric
        - Check integration with ROC
    """
    model_results_paired = list(zip(predicted_labels, test_labels))   # (model, Groundtruth) tuples    
    TN = model_results_paired.count((0, 0))
    TP = model_results_paired.count((1, 1))
    FP = model_results_paired.count((1, 0))
    FN = model_results_paired.count((0, 1))
    if TP>0 or TN>0:  
        TPR, precision, F, TNR, FDR = confusion_metrics(TP,TN,FN,FP)
    elif TP==0 and TN>0: 
        if verbose:
            print('\nNeed True Positives in order to compute metrics!\n')   
        TPR = precision = F = TNR = FDR = 0
    else:  
        TPR = precision = F = TNR = FDR = 0
    if verbose:
        print('\n\tRESULTS\n\t--------------\n\tRecall = %2.2f\n\tPrecision = %2.2f\n\tF1-value = %2.2f\n\tTNR = 1 - FPR = %2.2f\n' % (TPR, precision, F, TNR))
    return TPR, precision, F, TNR, FDR

"""
NOTA BENE: THE FOLLOWING FUNCTIONS, ALTHOUGH HOPEFULLY CORRECT, ARE NOT OPTIMAL COMPUTATIONALLY.
BETTER IS TO USE <<classification_plots.py>>
"""


def compute_AUC(TPR_values, FPR_values):
    """
    TO DO:  Check instructions (pseudo code) by <<Introduction to ROC analysis>> - Fawcett
    """
    return np.sum(-np.diff(FPR_values) * np.array(TPR_values[1:]))

def ROC_curve(predicted_probabilities, test_labels, verbose=True, plotting=True, N=100):
    """
    TO DO:  Check instructions (pseudo code) by <<Introduction to ROC analysis>> - Fawcett
    """
    TPRs = []
    FPRs = []
    # Random prediction probs for comparison:
    Random_Probs = np.random.rand(len(test_labels),)
    TPRs_rand = []
    FPRs_rand = []
    for prob_threshold in np.linspace(0,1,N):
        # actual predictions:
        predicted_labels = (predicted_probabilities > prob_threshold) * 1
        TPR, _, _, TNR, _ = binary_classifier_administration(predicted_labels, test_labels)
        TPRs.append(TPR)
        FPRs.append(1 - TNR)
        # random predictions:
        y_predict_Random = (Random_Probs > prob_threshold) * 1
        TPR, _, _, TNR, _ = binary_classifier_administration(y_predict_Random, test_labels)
        TPRs_rand.append(TPR)
        FPRs_rand.append(1 - TNR)
    if plotting:
        plt.figure(figsize=(9,8))
        plt.plot([0,1], [1,1], 'k:', alpha=.5)
        plt.plot([1,1], [0,1], 'k:', alpha=.5)
        plt.plot([0,0], [0,1], 'k:', alpha=.5)
        plt.plot([0,1], [0,0], 'k:', alpha=.5)
        plt.plot([0, 1], [0, 1], 'k--', alpha=.7)
        plt.fill_between(FPRs, TPRs, color='b', alpha=.2)
        plt.plot(FPRs, TPRs, 'b-', label='clf')
        plt.plot(FPRs_rand, TPRs_rand, 'r-', label='random')
        plt.xlabel('FPR', fontsize=14)
        plt.ylabel('TPR', fontsize=14)
        plt.title('ROC', fontsize=15)
        plt.legend(loc=4, fontsize=14)
    if verbose:  # compute AUC
        print('\nAUC = %.3f' % (compute_AUC(TPRs, FPRs)))
        print('AUC (random) = %.3f\n' % (compute_AUC(TPRs_rand, FPRs_rand)))
    #return TPRs, FPRs

def Precision_Recall_curve(predicted_probabilities, test_labels, verbose=True, plotting=True, N=100):
    Recall = []
    Precision = []
    # Random prediction probs for comparison:
    Random_Probs = np.random.rand(len(test_labels),)
    Recall_rand = []
    Precision_rand = []
    for prob_threshold in np.linspace(0,1,N):
        # actual predictions:
        predicted_labels = (predicted_probabilities > prob_threshold) * 1
        TPR, P, _, _, _ = binary_classifier_administration(predicted_labels, test_labels)
        Recall.append(TPR)
        Precision.append(P)
        # random predictions:
        y_predict_Random = (Random_Probs > prob_threshold) * 1
        TPR, P, _, _, _ = binary_classifier_administration(y_predict_Random, test_labels)
        Recall_rand.append(TPR)
        Precision_rand.append(P)
    if verbose:
        R = np.array(Recall)
        print('\nMaximum Precision is %.3f, reached at Recall of %.3f' % (max(Precision), R[np.nanargmax(Precision)]))
    if plotting:
        plt.figure(figsize=(9,8))
#        plt.plot([0,1], [1,1], 'k:', alpha=.5)
#        plt.plot([1,1], [0,1], 'k:', alpha=.5)
#        plt.plot([0,0], [0,1], 'k:', alpha=.5)
#        plt.plot([0,1], [0,0], 'k:', alpha=.5)
#        plt.plot([0, 1], [1, 0], 'k--', alpha=.7)
        plt.plot(Recall, Precision, 'g-', label='clf')
        plt.plot(Recall_rand, Precision_rand, 'r-', label='random')
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Recall-Precision curve for linear classifier', fontsize=15)
        plt.legend(loc=4, fontsize=14)

def False_Discovery_ratio(predicted_probabilities, test_labels, verbose=True, plotting=True, N=100):
    Recall = []
    probs = []
    FDR = []
    for prob_threshold in np.linspace(0,1,N):
        predicted_labels = (predicted_probabilities > prob_threshold) * 1
        R, _, _, _, FDRatio = binary_classifier_administration(predicted_labels, test_labels)
        probs.append(prob_threshold)
        Recall.append(R)
        FDR.append(FDRatio)
    if verbose:
        Recall = np.array(Recall)
        print('\nBest False Discovery Rate: %i FPs for each TP, reached at Recall of %.3f.' % (min(FDR), Recall[np.nanargmin(FDR)]))
    if plotting:
        plt.figure(figsize=(10,6))
        plt.plot(Recall, FDR, 'm-')
        #plt.xlabel('probability threshold', fontsize=14)
        plt.xlabel('TPR = Recall', fontsize=14)
        plt.ylabel('FP : TP', fontsize=14)
        plt.title('False Discovery Ratio for linear classifier', fontsize=15)
        plt.ylim(0, 50)

#if __name__=='__main__':
    # Example dummy data:
#    model_output = np.random.randint(0,2,100)
#    groundtruth = np.random.randint(0,2,100)
    
    # Compute Recall, Precision & F1:
#    Recall, Precision, F1 = binary_classifier_administration(model_output, groundtruth, verbose=True)
    
