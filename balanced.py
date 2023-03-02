
############ BALANCED CLUSTERS ###########

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.stats import invgamma

from functions_H import *


eta = 1
a = b = 1

N = 200
D = 7
Q = 4
C_q = int(N / Q)

num_datasets = 10


def xVector():
    x_1 = np.arange(D)
    x_cluster = x_1
    for i in range(C_q - 1):
        x_cluster = np.append(x_cluster, x_1)
    x_col = x_cluster.reshape((-1, 1))
    return x_col


def drawBeta_q(sigma2_q, k):
    cov_beta_q = sigma2_q * eta * np.eye(k)
    beta_q = np.random.multivariate_normal(np.repeat(0, k), cov_beta_q)
    return beta_q


def buildG_q(M_q, rho_q):
    A_q = eta * M_q

    R_1 = toeplitz(np.append(1, np.repeat(rho_q, D - 1)))
    R_q = np.zeros([D * C_q, D * C_q])
    for i in range(0, D * C_q, D):
        for j in range(0, D * C_q, D):
            if i == j:
                R_q[i:(i + D), i:(i + D)] += R_1

    G_q = A_q + R_q
    return G_q


def simulationOneCluster(sigma2_q, rho_q, M_q, k_q):
    beta_q = drawBeta_q(sigma2_q, k_q)
    G_q = buildG_q(M_q, rho_q)
    Y_q = np.random.multivariate_normal(np.zeros(D * C_q), G_q)
    e_q = np.random.normal(0, 1, k_q)
    Y_q = Y_q + e_q
    return Y_q





x_q = xVector()
M_q = poly_kernel(x_q, 2)
k_q = M_q.shape[1]


def createOneDataset(d):
    x = np.empty(0)
    beta = np.empty(0)
    y = np.empty(0)

    for q in range(Q):
        sigma2_q = invgamma.rvs(a, b)
        rho_q = np.random.uniform(0, 1, 1)

        beta_q = drawBeta_q(sigma2_q, k_q)
        y_q = simulationOneCluster(sigma2_q, rho_q, M_q, k_q)

        x = np.append(x, x_q)
        beta = np.append(beta, beta_q)
        y = np.append(y, y_q)

    id = np.repeat(np.arange(N), D)
    dataset = pd.DataFrame({'RID': id, 'Month': x, 'ADAS11': y})
    filename = f'balanced_{d}.csv'
    dataset.to_csv(filename, index = False)



for d in range(num_datasets):
    createOneDataset(d)
