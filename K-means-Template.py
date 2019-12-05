# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:39:52 2019

@author: DIPAK
"""

import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt


data = pd.read_csv('clustering.csv')
data.head()

X = data[["LoanAmount", "ApplicantIncome"]]


#number of Clusters
K =3
Centroids = (X.sample(n=K))
plt.scatter(X["ApplicantIncome"], X["LoanAmount"], c= 'black')
plt.scatter(Centroids["ApplicantIncome"], Centroids["LoanAmount"],c = 'red')
plt.xlabel('AnnualIncome')
plt.ylabel('Loan Amount (In Thousands)')
plt.show()



