
# coding: utf-8

# In[1]:

from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')


# ### 1D

# In[2]:

isolation_forest = IsolationForest()

data = np.concatenate((np.random.normal(size=100), np.random.normal(loc=5., size=100)))
isolation_forest.fit(data.reshape(-1, 1))

xx = np.linspace(-4, 10, 1000)
plt.plot(xx, isolation_forest.decision_function(xx.reshape(-1,1)))
plt.hist(data, normed=True)


# ### 2D

# In[67]:

X = np.random.randn(8000,2)


# In[68]:

isolation_forest = IsolationForest(n_estimators=15)
isolation_forest.fit(X)


# In[70]:

# test_pts = np.array([[0., .1], [3., -3.], [-2., 1.]])

xx, yy = np.meshgrid(np.linspace(-4, 4, 75), np.linspace(-4, 4, 75))
test_pts = np.c_[xx.ravel(), yy.ravel()]

# test_pts = np.random.randn(5,12)

# print isolation_forest.decision_function(test_pt)  # gives range: lower value = more likely to be outlier
P = isolation_forest.predict(test_pts)
P[P==-1] = 0  # these are the outliers.

plt_colors = ['ro', 'bo']
plt.figure(figsize=(10,10))
for i,tp in enumerate(test_pts):
    plt.plot(tp[0], tp[1], plt_colors[P[i]], alpha=.3)
#     if P[i]==1:
#         plt.plot(tp[0], tp[1], plt_colors[0], alpha=.1)
#     elif P[i]:
#         plt.plot(tp[0], tp[1], plt_colors[1], alpha=.1)
        
plt.plot(X[:,0], X[:,1], 'k.', markersize=6, alpha=.3)


# In[64]:




# In[ ]:



