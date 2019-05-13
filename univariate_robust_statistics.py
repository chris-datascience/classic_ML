# -*- coding: utf-8 -*-
"""
Created on Sun Jul 02 16:13:35 2017

@author: Kris
"""

import numpy as np
import matplotlib.pyplot as plt

def Median_of_Absolute_Deviations(population):
    pop = np.array(population)
    med = np.median(pop)
    return np.median(abs(pop - med))

def robust_z_scores(population):
    pop = np.array(population)
    return (pop - np.median(pop)) / Median_of_Absolute_Deviations(pop)
    
def create_univariate(n_points_max, n_distributions_max, spread_factor_max):
    n_distributions = np.random.randint(1, n_distributions_max+1)
    n_points_per_distribution = np.random.randint(1, n_points_max, size=(n_distributions))
    samples = []
    for i in range(n_distributions):
        samples.append(list(np.random.randint(low=1, high=spread_factor_max) * np.random.rand(n_points_per_distribution[i])))
    return np.array(sum(samples, []))

def probability_of_max_z_scores(n_simulations=7500, z_score_max=5, n_distributions=3, pop_size=100):
    critical_z_scores = np.arange(1, z_score_max+.5, .5)
    z_scores = []
    mean_probabilities = []
    for it, z in enumerate(critical_z_scores):
        probs = []
        for i in range(n_simulations):
            P = create_univariate(pop_size, n_distributions, n_distributions+2)
            n_greater_than_z = (robust_z_scores(P)>z).sum()
            prob_exceedance = 100.*float(n_greater_than_z) / len(P)
            probs.append(prob_exceedance)
        z_scores.append(z)
        mean_probabilities.append(np.mean(probs))
    return z_scores, mean_probabilities
    
        
if __name__=='__main__':
    
    # -Show simple univariate population and median + 2.5 * MAD-cutoff
    population = create_univariate(100, 3, 3)
    median = np.median(population)
    MAD = Median_of_Absolute_Deviations(population)

    plt.hist(population, bins='auto', alpha=.7)
    plt.plot([median]*2, [0,30], 'r-', lw=3, alpha=.9)
    plt.plot([median + MAD]*2, [0,30], 'r--', lw=3, alpha=.9)
    plt.plot([median + 2.5*MAD]*2, [0,30], 'g--', lw=3, alpha=.9)
    plt.title('Z_max = %1.2f'%np.max(robust_z_scores(population)), fontsize=15)
    plt.xlabel('value')
    plt.ylabel('frequency')

    plt_colors = ['bo-','ro-','co-','mo-','go-','yo-','ko-']

    
    # -Sensitivity on number of separate distributions in population-
    plt.figure(figsize=(8,6))
    for nd in range(1,6):
        z, mp = probability_of_max_z_scores(n_distributions=nd)
        plt.plot(z, mp, plt_colors[nd-1], label='n_dist='+str(nd))
    plt.xlabel('cut off z-score',fontsize=14)
    plt.ylabel('probability of occurence (%)',fontsize=14)
    plt.title('Impact number of distributions in population on outlier test', fontsize=16)
    plt.legend(fontsize=14)
    
    # -Sensitivity on population size-
    plt.figure(figsize=(8,6))
    for ps in range(1,4):
        pop_size = int(10**ps)
        z, mp = probability_of_max_z_scores(pop_size=pop_size)
        plt.plot(z, mp, plt_colors[ps-1], label='pop_size='+str(pop_size))
    plt.xlabel('cut off z-score',fontsize=14)
    plt.ylabel('probability of occurence (%)',fontsize=14)
    plt.title('Impact of population size on outlier test (n_dist=2)', fontsize=16)
    plt.legend(fontsize=14)