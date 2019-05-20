from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(1612838)
from sklearn.svm import SVC

means = [[4, 2], [4, 1]]
cov = [[.4, .1], [.1, .2]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N) # class 1
X1 = np.random.multivariate_normal(means[1], cov, N) # class -1 

print([(x1,x2) for x1,x2 in X0], [(y1,y2) for y1,y2 in X1])

X = np.concatenate((X0.T, X1.T), axis = 1) # all data 
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # labels 

y1 = y.reshape((2*N,))
X1 = X.T # each sample is one row
clf = SVC(kernel = 'linear', C = 100) # just a big number 

clf.fit(X1, y1) 

w = clf.coef_
b = clf.intercept_


print('w = ', w)
print('b = ', b)