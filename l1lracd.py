#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Obtain L1 norm low rank approximation. This code uses and Python version of the MATLAB code provided
by Nicolas Gillis, Component-Wise l1-Norm Low-Rank Matrix Approximation  
https://sites.google.com/site/nicolasgillis/code?authuser=0

Cite: 

"""

__author__ = 'Hazan Daglayan'
__all__ = ['L1LRAcd', 'wmedian']
from scipy.sparse import linalg
import numpy as np
import numpy.matlib


def L1LRAcd(M, r=1, maxiter=100, U0=None, V0=None):
    m, n = M.shape
    if U0 == None or V0 == None:
        U_, S_, VT_ = np.linalg.svd(M,full_matrices=0)
        V = VT_[:r]       # we cut projection matrix according to the # of PCs
        U = np.dot(U_[:, :r],np.diag(S_[:r]))


    else:
        U = U0
        V = V0
    

    for i in range(maxiter):
        R = M-np.dot(U,V)
        for k in range(r):
            # Current residue
            R = R + np.dot(np.reshape(U[:,k],(len(U[:,k]),1)),np.reshape(V[k,:],(1,len(V[k,:]))))
            # Weighted median subproblems
            U[:,k] = wmedian(R,V[k,:].T)
            V[k,:] = wmedian(R.T, U[:,k])
            # Update total residue
            R = R - np.dot(np.reshape(U[:,k],(len(U[:,k]),1)),np.reshape(V[k,:],(1,len(V[k,:]))))

    return U, V

def wmedian(A,y):

    ''' WMEDIAN computes an optimal solution of
    min_x  || A - xy^T ||_1 
    
    where A has dimension (m x n), x (m) and y (n),
    in O(mn log(n)) operations. Note that it can be done in O(mn). 
    
    This code comes from the paper 
    "Dimensionality Reduction, Classification, and Spectral Mixture Analysis 
    using Nonnegative Underapproximation", N. Gillis and R.J. Plemmons,
    Optical Engineering 50, 027001, February 2011.
    Available on http://sites.google.com/site/nicolasgillis/code'''

    # Reduce the problem for nonzero entries of y
    indi = np.absolute(y) > 1e-16
    A = A[:,indi]
    y = y[indi]
    m,n = A.shape
    A = A/y.T
    y = np.absolute(y)/np.sum(np.absolute(y))
    
    
    # Sort rows of A, m*O(n log(n)) operations
    Inds = np.argsort(A)
    As = np.take_along_axis(A, Inds, axis=1)
    Y = y[Inds]
    # Extract the median
    actind = np.arange(m)
    i = 0; 
    sumY = np.zeros((m,1))
    x = np.zeros((m,1))
    
    while len(actind):
        #sum of the weights
        sumY[actind] = sumY[actind] + np.reshape(Y[actind,i], (len(Y[actind,i]),1))
        # check which weight >=0
        supind = sumY[actind,:].reshape(-1) >=0.5
        # update corresponding x
        x[actind[supind],0] = As[actind[supind],i]
        # only look reminding x to update
        actind = actind[~supind]
        i = i+1
    
    return x.reshape(-1)
    
    
    
    
    
    
    
    