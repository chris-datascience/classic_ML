# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 22:29:52 2019

@author: Kris
"""

import pandas as pd
import numpy as np

# [0] Standard error of standard deviation of normal distribution

# [1] Comparing categorical variable distributions via contingency table
# Example appears in <<Fraud Analytics>> by Baesens et al. (Wiley)
product_counts = {'clothes':(1000,500), 'books':(2000,100), 'music':(3000,200), 'electro':(100,80), \
                  'games':(5000,800)}
transactions = pd.DataFrame(columns=['fraud_flag', 'product'])
for prod,cnts in product_counts.items():
    Prod_txs = pd.DataFrame(np.random.permutation(['no_fraud']*cnts[0] + ['fraud']*cnts[1]), columns=['fraud_flag'])
    Prod_txs['product'] = prod
    transactions = pd.concat([transactions, Prod_txs], axis=0)
transactions['dummy'] = 1
transactions.index = np.random.permutation(range(len(transactions)))
transactions.index.name = 'customer_id'
transactions.sort_index(inplace=True)

# Pivot/Group by product
txs_pivot = pd.pivot_table(transactions, index=['fraud_flag'], columns=['product'], values='dummy', aggfunc=np.sum)
txs_pivot['total'] = txs_pivot.sum(axis=1)
txs_pivot_T = txs_pivot.T
txs_pivot_T['odds'] = txs_pivot_T['no_fraud'] / txs_pivot_T['fraud']
txs_pivot = txs_pivot_T.T
txs_pivot.loc['total'] = list(txs_pivot[:2].sum(axis=0).values)
print('\nEmpirical Frequencies all groups:')
print(txs_pivot)

# Now create product classes in two different ways
Classes1 = txs_pivot[:2].copy()
Classes1['books_and_music'] = Classes1['books'] + Classes1['music']
Classes1['others'] = Classes1['electro'] + Classes1['games']
Classes1 = Classes1[['clothes', 'books_and_music', 'others']]
Classes1['total'] = Classes1.sum(axis=1).values
Classes1.loc['total'] = list(Classes1.sum(axis=0).values)
print('\nEmpirical Frequencies grouped:')
print(Classes1)

# Compute Independent Frequencies
