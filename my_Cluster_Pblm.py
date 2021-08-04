import pymanopt as mo
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import SteepestDescent
import math
from random import seed
from random import randint
# seed random number generator
seed(1)

from Y_mani import Y_mani
from random_creators import dist_mat

from sklearn.cluster import KMeans

import numpy as np
import numpy.linalg as la

import itertools
import collections


class Cluster_Pblm:

    def __init__(self, P, k):
        """
        P is a (d x n) matrix holding our points.
        k is the number of clusters expected
        If testing is true then we do extra computations which are useful but not computationally necessary.
        """
        self.P = P  # Matrix of points
        self.data = np.transpose(self.P)
        self.nu = np.sum(P ** 2, axis=0)[:, None]  # Matrix of pts' norms.
        self.d = P.shape[0]  # Dimension the points lie in
        self.n = P.shape[1]  # number of points
        self.k = k  # number of clusters
        self.M = Y_mani(self.n, self.k)  # "Y" manifold corresponding to our problem.
        # NB it may be quicker to use "sums" and such strictly instead of the ones matrix
        self.one = np.ones((self.n, 1))  # Matrix of all ones.

    def tr(self, Y):
        """
        Returns tr(DYY^T)
        This code uses the computation which is linear in n.
        """
        nu, P, one = self.nu, self.P, self.one
        term1 = 2 * ((one.T.dot(Y)).dot(Y.T.dot(nu)))[0, 0]
        term2 = -2 * np.sum(P.dot(Y) ** 2)
        return (term1 + term2)

    def gr_tr(self, Y):
        """
        Returns the (euclidean) gradient of tr
        This code uses the computation which is linear in n.
        """
        nu, P, one = self.nu, self.P, self.one
        return (2 * (one.dot(nu.T.dot(Y))
                     + nu.dot(one.T.dot(Y)))
                - 4 * P.T.dot(P.dot(Y)))

    def gr_tr_projected(self, Y):
        """
        Returns the M gradient of tr
        """
        W = self.gr_tr(Y)
        return self.M.proj(Y, W)

    def neg(self, Y):
        """
        Returns the norm of the negative part of Y.
        """
        negpt = Y * (Y < 0)
        return (negpt ** 2).sum()

    def gr_neg(self, Y):
        """
        Returns the (euclidean) gradient of the negative part of Y
        """
        return 2 * Y * (Y < 0)

    def fn_weighted(self, a, b):
        """
        Returns a function which computes
        a*neg(Y) + b*tr(Y)
        """
        return lambda Y: a * self.tr(Y) + b * self.neg(Y)

    def gr_weighted(self, a, b):
        """
        Returns a function which computes
        a*gr_neg(Y) + b*gr_tr(Y)
        """
        return lambda Y: a * self.gr_tr(Y) + b * self.gr_neg(Y)

    def run_minimization(self, Y, gamma=pow(10, -2), eta=pow(10, -4), rho=6*pow(10, 2)):

        X = Y
        Z = Y
        Lambda = np.zeros((self.n, self.k))
        # Calculating D matrix
        Y_log = [Y]
        X_log = [X]

        # Find D matrix
        D = np.zeros((self.n, self.n))
        for i in range(self.n):
            #print(self.data[i])
            for j in range(self.n):
                D[i][j] = np.linalg.norm([self.data[i] - self.data[j]], 2)**2

        # Run Algorithm
        for i in range(1000):
            for _ in range(15):
                X = self.M.retr_fast(X, -eta * self.M.proj_fast(X, 2 * D@X + (Lambda + rho * (X - Z))))
            #print(np.linalg.norm(X.T.dot(X) - np.eye(self.k)))
            #print(np.linalg.norm(X.dot(X.T).dot(np.ones(self.n)) - np.ones(self.n)))
            #Y = np.where(Y < 0, 0, X + (1 / rho) * Lambda)
            Y = np.maximum(0, X + (1 / rho) * Lambda)
            Z = (1 / ((1 / gamma) + rho)) * ((1 / gamma) * Y + Lambda + rho * X)
            Lambda = Lambda + rho * (X - Z)
            if (i+1) % 10 == 0:
                print(f'Trace {i+1}: {self.tr(X)}, Norm: {np.linalg.norm(X * (X < 0))}')
            # X_log.append(X)
            # Y_log.append(Y)
        return X


    def find_initial(self):
        Y0 = np.zeros((self.n, self.k))
        for i in range(self.n):
            Y0[i][randint(0,self.k-1)] = 1
        Y0 = Y0 / np.transpose(np.sqrt(np.diag(np.transpose(Y0)@Y0)))
        return Y0


    def do_path(self, gamma=pow(10, -2), eta=pow(10, -4), rho=6*pow(10, 2)):
        #Y0 = self.find_initial()
        Y0 = -np.array(self.M.rand())
        Ys = self.run_minimization(Y0, gamma=pow(10, -2), eta=pow(10, -4), rho=6*pow(10, 2))
        print(len(Ys), "x", len(Ys[0]))
        return Ys