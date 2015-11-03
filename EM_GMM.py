# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 10:27:48 2015

@author: mik
"""

#X has each example in row
#

import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn import mixture
import pdb
#everything in np.array
#X is NxM array and has exmaples in rows. Mu is KxM has mu's in rows. Sigmas  
#is a KxMxM array. Pi is a K-by-1 array.
   
def log_prob(X, Mu , Sigma_inv):
    def k_th_col_pmtrx(X, mu, sigma_inv):
        X_centered = X - mu        
        return (-0.5) * np.sum(X_centered.dot(sigma_inv) * X_centered, \
                                                                      axis = 1)
    return np.array([k_th_col_pmtrx(X, Mu[k, :], Sigma_inv[k, :, :]) \
    for k in range(0, Mu.shape[0])]).T
    
def pmtrx(X, Mu, Sigma):
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    return (1. / np.sqrt(2 * np.pi) ** X.shape[1]) * \
    (np.exp(log_prob(X, Mu, Sigma)) / np.sqrt(abs(Sigma_det))) 
    
def EM_GMM(X, Mu, Sigma, Pi, num_iter):
   # pdb.set_trace()    
    N = X.shape[0]
    M = X.shape[1]
    K = Pi.shape[0]
    
    for t in range(0, num_iter):
#E step        
        P = pmtrx(X, Mu, Sigma)
        R = (1. / P.dot(Pi)).dot(Pi.T) * P
#M Step
        Pi_new = np.mean(R, axis = 0).reshape(K, 1)        
        Mu_new = (1. / N) * ((X.T).dot(R) * (1. / Pi_new.T)).T
        Sigma_new = np.array([ \
        (1. / (Pi_new[k][0] * N)) * \
        (R[:, k] * X.T).dot(X) - \
        (Mu_new[k, :].reshape(K, 1)).dot(Mu_new[k, :].reshape(1, K)) \
        for k in range(0, K)])
#Update
        Mu = Mu_new
        Sigma = Sigma_new
        Pi = Pi_new          
    result = [Mu, Sigma, Pi]
    return result 

def get_synthetic_data(Mu, Sigma, Pi, num_samples = 1000):
    components = np.random.RandomState(1).choice(Pi.shape[0], num_samples, 
                                                          p = Pi.T.ravel())
    Sigma_sqrtm = np.array([scipy.linalg.sqrtm(Sigma[k]) \
    for k in range(0, Sigma.shape[0])])
    unit_gaussian_samples = np.random.RandomState(1).randn(num_samples, 2)
    actual_samples = np.array([sample.dot(Sigma_sqrtm[c, :, :]) + Mu[c] \
    for sample, c in zip(unit_gaussian_samples, components)])
    return actual_samples 

def test (num_samples, num_iter, Mu, Sigma, Pi, Mu_0, Sigma_0, Pi_0):
    X = get_synthetic_data(Mu, Sigma, Pi, num_samples)
    plt.scatter(X[:, 0], X[:, 1], alpha = 0.1)
    plt.show()
    Mu_res , Sigma_res, Pi_res = EM_GMM(X, Mu_0, Sigma_0, Pi_0, num_iter)
    X_res = get_synthetic_data(Mu_res, Sigma_res, Pi_res, num_samples)
    plt.scatter(X_res[:, 0], X_res[:, 1], alpha = 0.1)
    plt.show()
    return Mu_res , Sigma_res, Pi_res