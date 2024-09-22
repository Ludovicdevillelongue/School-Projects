# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:54:01 2020

@author: ludov
"""

import pandas as pd
from pandas import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import warnings





'-------------------------------// Yield Curves and PCA //-----------------------------'


warnings.filterwarnings('ignore')

#dataframe creation
file=ExcelFile(r"Morocco_Yield_Curves.xlsx")
df=file.parse('Data')
#monthly aggregation of the data in a dataframe "grouped" and calculation of the relative average for each month
date=df['TIME_PERIOD']
df['year_month'] = pd.to_datetime(df['TIME_PERIOD']).dt.to_period('M')
grouped = df.groupby('year_month').mean()
print('\n\n')

#yield curves with multiple dates

#choose years 2004, 2007, 2008, 2009, 2012, 2015, and 2018
selections=[2013,2014,2015,2016,2017,2018,2019]

#variable creation to sort indexes
m=grouped.index.month
m.tolist()
y=grouped.index.year
y.tolist()

#sort the month (December) and the desired years, retrieve the rates for these dates and create a chart according to maturity
plt.rcParams["figure.figsize"] = (30,12)
plt.figure(1)
for i in range(0,len(grouped)):
    for n in range(0,len(selections)):
        if m[i]==12 and y[i]==selections[n]:
            values=grouped.iloc[i, 0:len(grouped.columns)]
            values.tolist()
            headers=list(grouped.columns[0:len(grouped.columns)].values)
            plt.xlabel('maturities', fontsize=12)
            plt.ylabel('returns', fontsize=12)
            plt.plot(headers,values,label="returns "+str(m[i])+"-"+str(y[i]))
            plt.title('Yield curves for several years', fontsize=16)
            plt.legend(loc="best",fontsize=12)
plt.show()
        
#samples
sample1=grouped['2014-04':'2016-04']
sample2=grouped['2018-04':'2020-04']


#necessary functions
normalize=lambda x: (x-x.mean())/x.std()
fractions=lambda x: x/x.sum() 


#principal components in sample 1
pca1=PCA().fit(sample1.apply(normalize))
pca_components1=pca1.fit_transform(sample1)
pca1.explained_variance_ratio_

#principal components in sample 2
pca2=PCA().fit(sample2.apply(normalize))
pca_components2=pca2.fit_transform(sample2)
pca2.explained_variance_ratio_

#calculation of sample 1 eigenvectors
cov1=np.cov(sample1.T)
eigen_values, eigen_vectors = np.linalg.eig(cov1)
#representation of the first three eigenvectors
plt.figure(2)
plt.plot(sample1.columns, eigen_vectors.T[0], label="PCA_1")
plt.plot(sample1.columns, eigen_vectors.T[1],label="PCA_2")
plt.plot(sample1.columns, eigen_vectors.T[2],label="PCA_3")
plt.legend(loc='best', fontsize=12)
plt.xlabel('maturities', fontsize=12)
plt.ylabel('eigenvectors', fontsize=12)
plt.title('Principal Component Analysis of sample 1 eigenvectors', fontsize=16)
plt.rcParams["figure.figsize"] = (30,12)
plt.show()

#calculation of sample 2 eigenvectors
cov2=np.cov(sample2.T)
eigen_values, eigen_vectors = np.linalg.eig(cov2)
#representation of the first three eigenvectors
plt.figure(3)
plt.plot(sample2.columns, eigen_vectors.T[0], label="PCA_1")
plt.plot(sample2.columns, eigen_vectors.T[1],label="PCA_2")
plt.plot(sample2.columns, eigen_vectors.T[2],label="PCA_3")
plt.legend(loc='best', fontsize=12)
plt.xlabel('maturities', fontsize=12)
plt.ylabel('eigenvectors', fontsize=12)
plt.title('Principal Component Analysis of sample 2 eigenvectors', fontsize=16)
plt.rcParams["figure.figsize"] = (30,12)
plt.show()


#addition of the March 2020 yield curve in the chart containing all yield curves
yield_2020=plt.figure(4)
for i in range(0,len(grouped)):
    if m[i]==3 and y[i]==2020:
        values_mars2020=grouped.iloc[i, 0:len(grouped.columns)]
        values_mars2020.tolist()
        headers=list(grouped.columns[0:len(grouped.columns)].values)
        plt.plot(headers,values_mars2020,label="returns "+str(m[i])+"-"+str(y[i]))
        plt.legend(loc="best", fontsize=12)
plt.show()


