# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 10:27:48 2015

@author: mik
"""

#X has each example in row
#

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
# from sklearn import mixture

#X is NxM array and has exmaples in rows. Mu is KxM has mu's in rows. Sigmas  
#is a KxMxM array. Pi is a K-by-1 array.


def log_prob(X, Mu, Sigma_inv):
    '''
    X is a num_datapoints x data_dimension matrix
    Mu is a num_mixture_components x data_dimension matrix
    Sigma_inv is a num_mixture_components x data_dimension x data_dimension matrix
    Returns a num_datapoints x num_mixture_components matrix'''
    def k_th_col_pmtrx(X, mu, sigma_inv):
        X_centered = X - mu
        return (-0.5) * np.sum(X_centered.dot(sigma_inv) * X_centered, axis=1)
    return np.array([k_th_col_pmtrx(X, Mu[k, :], Sigma_inv[k, :, :]) for k in range(0, Mu.shape[0])]).T


def pmtrx_unnormalized(X, Mu, Sigma, epsilon=0.0):
    # TODO: test against performing Cholesky decomposition on Sigma: Cholesky is probaby faster
    Sigma_det = np.linalg.det(Sigma+np.identity(Sigma.shape[1])[None, :, :]*epsilon)
    Sigma_inv = np.linalg.inv(Sigma+np.identity(Sigma.shape[1])[None, :, :]*epsilon)
    return (np.exp(log_prob(X, Mu, Sigma_inv)) / np.sqrt(Sigma_det))


def pmtrx(X, Mu, Sigma, epsilon=0.0):
    return pmtrx_unnormalized(X, Mu, Sigma, epsilon=0.0) / (np.sqrt(2.*np.pi) ** X.shape[1])


def EM_GMM(X, Mu, Sigma, Pi, num_iter):
    N = X.shape[0]
    K = Pi.shape[0]

    for _ in range(0, num_iter):
        # E step
        R = pmtrx_unnormalized(X, Mu, Sigma)
        R /= np.sum(R, axis=1)[:, None]
        # M Step
        Pi_new = np.mean(R, axis=0)[:, None]
        Mu_new = R.T.dot(X) / (Pi_new * N)
        Sigma_new = np.array([
            (1. / (Pi_new[k, 0] * N)) *
            (R[:, k] * X.T).dot(X) -
            (Mu_new[k, :][:, None]).dot(Mu_new[k, :][None, :])
            for k in range(0, K)])
        # Update
        Mu = Mu_new
        Sigma = Sigma_new
        Pi = Pi_new
    result = [Mu, Sigma, Pi]
    return result


def get_synthetic_data(Mu, Sigma, Pi, num_samples=1000):
    components = np.random.RandomState(1).choice(Pi.shape[0], num_samples, p=Pi.T.ravel())
    Sigma_sqrtm = np.array([scipy.linalg.sqrtm(Sigma[k]) for k in range(0, Sigma.shape[0])])
    unit_gaussian_samples = np.random.RandomState(1).randn(num_samples, 2)
    actual_samples = np.array([sample.dot(Sigma_sqrtm[c, :, :]) + Mu[c] for sample, c in zip(unit_gaussian_samples, components)])
    return actual_samples


def test(num_samples, num_iter, Mu, Sigma, Pi, Mu_0, Sigma_0, Pi_0):
    X = get_synthetic_data(Mu, Sigma, Pi, num_samples)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.1)
    plt.show()
    Mu_res , Sigma_res, Pi_res = EM_GMM(X, Mu_0, Sigma_0, Pi_0, num_iter)
    X_res = get_synthetic_data(Mu_res, Sigma_res, Pi_res, num_samples)
    plt.scatter(X_res[:, 0], X_res[:, 1], alpha=0.1)
    plt.show()
    return Mu_res, Sigma_res, Pi_res


if __name__ == '__main__':
    Mu = np.array([[0., 0.], [10., 10.]])
    Sigma = np.array([[[1., 0.], [0., 1.]],
                      [[2., 1.], [1., 2.]]])
    Pi = np.array([[0.3], [0.7]])
    X = get_synthetic_data(Mu, Sigma, Pi, num_samples=4)
    Mu_res, Sigma_res, Pi_res = test(10000, 1000, Mu, Sigma, Pi,
                                     Mu_0=np.random.randn(2, 2),
                                     Sigma_0=np.array([np.identity(2) for _ in range(2)]),
                                     Pi_0=np.ones((2, 1))/2.)
    print Mu - Mu_res
    print Sigma - Sigma_res
    print Pi - Pi_res
