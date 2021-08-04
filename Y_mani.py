import pymanopt as mo
from pymanopt.manifolds.manifold import Manifold
import numpy as np
import numpy.linalg as la
import numpy.random as nrand
import scipy
import scipy.linalg as sla
import random_creators


def expbar(W):
    """
    For a square matrix W
    Returns (I - (1/2)*W^{-1})*(I + (1/2)*W)
    which is a approximation for exp(W) with some good algebraic properties.
    """
    I = np.eye(W.shape[0])
    return la.inv((I - W/2.0)).dot(I + W/2.0)

def expbar_quick(A,B,Y):
    """
    Suppose Y, A, and B are (n x k) with k << n
    Let W = AB.T - BA.T, which is n x n
    This function returns
    expbar(W)*Y, but computed in a quicker way which takes advantage of q << p.
    """
    n,k = Y.shape
    U = np.hstack((A,B))
    V = np.hstack((B,-A))
    inv = la.inv(np.eye(2*k,2*k) - (1/2)*V.T.dot(U))
    return Y + (U.dot(inv)).dot(V.T.dot(Y))


class Y_mani(Manifold):
    """
    pymanopt manifold holding representing
    \{ Y \in \real^{n \times k} : YY^T 1 = 1, Y^T Y = Id \}
    """
    def __init__(self, n, k):
        n, k = int(n), int(k)
        if(n <= 0 or k <= 0):
            raise TypeError("n and k must be positive integers.")
        self._n = n
        self._k = k
        self._name = "Y_mani with n="+str(n)+" and k="+str(k)+"."
        self.one = np.ones((n, 1))

    @property
    def typicaldist(self):
        """
        Not implemented in any true way"""
        return self._k/10
    
    def proj_slow(self, Y, W):
        """
        Project W onto the tangent space at Y using the naive slow computation
        """
        one = self.one
        x = (1/self._n)*W.dot(Y.T).dot(one)
        x1sym = x.dot(one.T) + one.dot(x.T)
        WYsym = W.T.dot(Y) + Y.T.dot(W)
        Omega = (1/4)*(WYsym - 2*la.multi_dot([Y.T, x1sym, Y]))
        V = W - 2*Y.dot(Omega) - x1sym.dot(Y)
        return V

    def proj_fast(self, Y, W):
        """
        Project W onto the tangent space at Y
        Uses clever order of matrix multiplication to make the computation linear in n.
        """
        z = sum(Y)[None,:] #z = one.T.dot(Y).  [None,:] makes it a (1,k) array.
        W_bar = W.T.dot(Y) #W_bar is k times k.
        Z = (1/self._n)*(W.dot(z.T.dot(z))
                         + self.one.dot(z.dot(W_bar)))
        Omega = ((W_bar + W_bar.T)/4 
                 - Y.T.dot(Z)/2)

        return W - 2*Y.dot(Omega) - Z

    proj = proj_fast
    egrad2rgrad = proj_fast

    def inner(self, Y, V1, V2):
        return (V1*V2).sum()
    
    
    def retr_slow(self, Y, V):
        """
        Perform the retraction of Y + proj(V) onto the manifold M
        Y is in M and V is a vector in ambient matrix space
        Returns a member of M.
        """
        V = self.proj_slow(Y,V)
        def equick(W):  #Quick approximation of exp with good algebraic properties.
            I = np.eye(self._n)
            return la.inv((I - W/2.0)).dot(I + W/2.0)
        Ap = Y.dot(Y.T).dot(V).dot(Y.T)
        B = V.dot(Y.T) - Y.dot(V.T) - 2*Ap
        Y_new = sla.expm(B).dot(sla.expm(Ap)).dot(Y)
        return Y_new
        
    def retr_fast(self, Y, V):
        """
        Perform the retraction of Y + proj(V) onto the manifold M
        Y is in M and V is a vector in ambient matrix space
        Returns a member of M.
        Uses a clever factorization of a rational approximation of the matrix exponential to make the computaiton linear in n.
        """
        V = self.proj_fast(Y,V)
        A = Y.T.dot(V)
        VL = V-Y.dot(A)
        Y_exp_A = Y.dot(expbar(A))
        Y_new = expbar_quick(VL, Y, Y_exp_A)
        return Y_new

    retr = retr_fast
        
    def norm(self, Y, V):
        return la.norm(V)

    def check_point(self, Y):
        eyek = np.eye(self._k)
        print(la.norm(Y.T.dot(Y) - eyek))

    def check_vector(self, Y, V):
        print(la.norm(self.proj(Y,V)-V))
        
    def rand(self):
        """
        Generates a point on M
        No real guarantee on the distribution.
        """
        Y =  random_creators.rand_basis_with1(self._n,self._k)
        return Y

    def round_clustering(self, Y):
        """
        Rounds the point Y to a clustering matrix.
        The assumption is that Y is already quite close to a clustering matrix.
        """
        #Create matrix which is 1 at the maximum of each row.
        max_locs = Y.argmax(axis=1) #The column ind of each maximum.
        maxes = np.zeros((self._n, self._k))
        maxes[np.arange(0, self._n, 1), Y.argmax(axis=1)] = 1 
        column_sizes = np.sqrt(maxes.sum(axis=0)) #Normalization factor for each col.
        return maxes/column_sizes
