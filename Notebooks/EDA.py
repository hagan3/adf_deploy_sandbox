#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

import pandas as pd

import sklearn
from sklearn import cluster
from sklearn import ensemble
from sklearn import feature_selection
from sklearn import metrics
from sklearn import model_selection
from sklearn import neural_network
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import r2_score
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from ipywidgets import widgets
from ipywidgets import HBox, Label
from IPython.display import HTML

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

import pdb

# import keras
# from keras import models
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM

import itertools
from itertools import permutations

import IPython
# print(IPython.sys_info())


import math
import time
import seaborn as sns

import fancyimpute
from fancyimpute import KNN

from scipy import stats

import warnings
warnings.filterwarnings('ignore')

from pandas_profiling import ProfileReport

print(pd.__version__)
print(np.__version__)


features_water_data_df = pd.read_csv(r"C:\Users\ryan.hagan\Documents\iwwgit\pi tox raw trimmed.csv", encoding = "ISO-8859-1")
reduced_water_data_df = pd.read_csv(r"C:\Users\ryan.hagan\Documents\iwwgit\features_filled_simple.csv", encoding = "ISO-8859-1")


# In[2]:


profile = ProfileReport(reduced_water_data_df, title='Water Profiling Report', explorative=True)


# In[5]:


# profile = ProfileReport(full_water_data_df, minimal=True)
# profile.to_file("full water data profile.html")


# In[5]:


profile.to_notebook_iframe()


# In[8]:


from autoviz.AutoViz_Class import AutoViz_Class

AV = AutoViz_Class()



# In[14]:


dft = AV.AutoViz(filename="none", sep="none", depVar="PercentDead", dfte=features_water_data_df, header=0, verbose=2, lowess=False, chart_format="svg", max_rows_analyzed=1500, max_cols_analyzed=30)


# In[ ]:




