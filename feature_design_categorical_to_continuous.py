# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:14:24 2018

@author: erdbrca
"""

import pandas as pd
import numpy as np


"""
Applying Bayes' theorem to derive numerical continuous features from discrete, categorical features.
This could work as an alternative for dummy one-hot encoding of categoricals in classification problems.
"""

# Suppose have this dataset:
X1 = pd.DataFrame(columns=['label'], data=np.random.randint(0,2,size=20))
X1['feature1'] = list('ABC'*6 + 'AA')
X1['feature2'] = list('XXYY'*5)
X1['dummy'] = 'null'

"""
Then can apply Bayes' rule as follows.
We want to design two new features, and do this by computing chance of label being equal to one given a certain feature value.

Feature1:
    P(label=1 | feature1='A') = P(feature1='A' | label=1) * P(label=1) / evidence_term

P(label=1) is universal: (X1.label==1).mean()
The denominator or 'evidence' is the same for all feature values so doesnt need to be computed.
And so is the prior! (chance of label being 1) so we leave that out too.

The likelihood P(feature1='A' | label=1) is computed as ((X1.label==1) & (X1.feature1=='A')).sum() / (X1.label==1).sum()
Can do this in a more vectorised way for feature1 thus:
F1 = pd.DataFrame(X1.feature1[X1.label==1].value_counts())
F1['feature1'] = F1.feature1 / F1.feature1.sum()
X1 = pd.merge(X1, F1, how='left', left_on=['feature1'], right_index=True)

And to do this for all features, see code directly below:
"""
X = X1.copy() 
for f in ['feature1','feature2']:
    F = pd.DataFrame(X.loc[X.label==1, f].value_counts())
    F[f] /= F[f].sum()
    X = pd.merge(X, F, how='left', left_on=[f], right_index=True)
X_contd = X[['label'] + [col for col in X.columns if col.endswith('_y')]].copy()

print('\nOriginal features:\n', X[['label','feature1_x','feature2_x']])
print('\nContinuous features:\n', X_contd)


# --------------------------------------------------------------------------------------------------------------
# Can use this function in practice:
def from_categoricals_to_continuous_features(df, feature_columns, label_column='label'):
    """
    Assume we have binary features and interested in likelihoods of label==1.
    See feature_design_categorical_to_continuous.py
    For inplace conversion!
    """
    X = df.copy()
    if X.isnull().sum().sum()>0:
        print('\nDrop NaNs from dataframe before converting to new features!\n')
        return None
    for f in feature_columns:
        F = pd.DataFrame(X.loc[X.label==1, f].value_counts())
        F[f] /= F[f].sum()
        X = pd.merge(X, F, how='left', left_on=[f], right_index=True)
    X_contd = X[[col for col in X.columns if not col.endswith('_x')]].copy()
    return X_contd

XX = from_categoricals_to_continuous_features(X1, ['feature1','feature2'], 'label')
# --------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------
#   ALTERNATIVELY, create a dictionary of likelihood dataframes for each features, to be applied to another set later:
# ------------------------------------------------------------------------------------------------------
def train_feature_likelihoods(df, feature_columns, label_column='label'):
    """
    Assume we have binary features and interested in likelihoods of label==1.
    See feature_design_categorical_to_continuous.py
    """
    X = df.copy()
    if X.isnull().sum().sum()>0:
        print('\nDrop NaNs from dataframe before converting to new features!\n')
        return None
    likelihoods = {}
    for f in feature_columns:
        F = pd.DataFrame(X.loc[X[label_column]==1, f].value_counts())
        F[f] /= F[f].sum()
        likelihoods[f] = F.copy()
    return likelihoods

# N.B. Later, we'll need to apply this to a set with identical features with similar values as follows:
def apply_feature_likelihoods(df, feature_columns, likelihoods_dict, fill_nans=True):
    X = df.copy()
    for f in feature_columns:
        try:
            F = likelihoods_dict[f]
            X = pd.merge(X, F, how='left', left_on=[f], right_index=True)
        except KeyError as missing_feature:
            print('feature %s not in likelihood dict' % missing_feature)
        except Exception as e:
            print('General exception occurred, namely: ', e)
    print(X.columns)
    X_contd = X[[col for col in X.columns if not col.endswith('_x')]].copy()
    if fill_nans:
        X_contd = X_contd.fillna(X_contd.mean())  # fills missing feature values with mean of that feature
    return X_contd
    
# Application example:
L_dict = train_feature_likelihoods(X1, ['feature1','feature2'], 'label')
# Now create a new df (without labels!) to apply this to:
Y = pd.DataFrame(columns=['feature1'], data=list('ABC'*6 + 'AA'))
Y['feature2'] = list('XXYYZ'*4)  # N.B. value Z not in training set
Y['dummy'] = 'blabla'
Y2 = apply_feature_likelihoods(Y, ['feature1','feature2'], L_dict)

