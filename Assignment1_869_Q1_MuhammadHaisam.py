#!/usr/bin/env python
# coding: utf-8

# In[51]:


# [Muhammad, Haisam]
# [20195819]
# [MMA]
# [Winter]
# [869]
# [12-08-2020]

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 
import pandas_profiling
import scipy
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# This will ensure that matplotlib figures don't get cut off when saving with savefig()
rcParams.update({'figure.autolayout': True})
import cufflinks as cf
get_ipython().run_line_magic('matplotlib', 'inline')


# In[52]:


# Answer to Question [1], Part [a]


# In[53]:


#loading the data
df = pd.read_csv('jewelry_customers.csv')


# In[54]:


list(df)
df.shape
df.info()
df.describe().transpose()
df.head(n=20)
df.tail()


# In[55]:


#Checking if the data has any missing values
df.isna().sum()


# In[56]:


# Answer to Question [1], Part [b]


# In[58]:


#Using pycaret to preprocess the data for clustering algorithms 


# In[57]:


#import clustering module
from pycaret.clustering import *

#intialize the setup
clu1 = setup(df, remove_multicollinearity = True, multicollinearity_threshold = 0.9, 
             session_id=123, log_experiment=True, log_plots = True, 
             transformation=True,
             pca=True,pca_components=0.99,pca_method='linear',
             ignore_low_variance = True, group_features = ['Income','Savings','Age'] )
             


# In[59]:


# creating k-means model
kmeans = create_model('kmeans')

# assign labels using trained model
kmeans_df = assign_model(kmeans)


# In[60]:


plot_model(model = kmeans, plot = 'cluster')
plot_model(model = kmeans, plot = 'elbow')
plot_model(model = kmeans, plot = 'silhouette')
plot_model(kmeans, plot = 'distribution') #to see size of clusters


# In[61]:


#Using k = 5 as per the k-elbow visualizer 

kmeans = create_model('kmeans', num_clusters = 5)


# In[62]:


kmeans_results = assign_model(kmeans)
kmeans_results.head()
kmeans_results['Cluster'].value_counts()


# In[63]:


# Answer to Question [1], Part [c]


# In[64]:


kmeans_results.groupby('Cluster').describe().transpose()


# In[66]:


print(kmeans)

