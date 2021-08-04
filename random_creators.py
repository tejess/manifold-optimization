import numpy as np
import numpy.linalg as la
import numpy.random as nrand
import scipy.linalg as sla
import scipy.sparse.linalg as spla
import autograd.numpy as anp

from pymanopt.manifolds import Stiefel

def antisym(mat):
    """Project mat onto antisymmetric matrices"""
    return (mat.T - mat)/2

def rand_basis_with1(n, k):
    """Return a random o.n. basis containing the ones vector"""
    one = anp.ones((n, 1))
    stief = Stiefel(n, k-1) #Gives us a basis with 
    Z = stief.rand()
    Y = sla.orth(anp.hstack((one, Z)))
    return Y

def rand_basis(n, k):
    return Stiefel(n, k).rand()

def disjoint_basis(spts):
    """
    spts is a list of nats n_1 n_2 ... n_k
    Returns the basis consisting of the vectors with 1/\sqrt{n_i} in n_i spots, disjointly
    """
    n = sum(spts)
    k = len(spts)
    starts = np.cumsum(spts)
    starts = np.insert(starts, 0, 0)
    def vec(i):
        x = np.vstack([np.zeros((starts[i], 1)),
                       np.ones((spts[i], 1)),
                       np.zeros((n-starts[i+1], 1))
                   ])
        x = x/np.sqrt(spts[i])
        return x
    return np.hstack([vec(i) for i in range(k)])

def perturb_rand_SOn(Y0, delta):
    n = Y0.shape[0]
    #omega = antisym(nrand.random((n,n)))
    omega = antisym(nrand.normal(size=(n,n)))
    omega = (delta/la.norm(omega))*omega
    Q = sla.expm(omega)
    return Q.dot(Y0)

def SOn_trail(Y0, omega, deltas):
    n = Y0.shape[0]
    omega = antisym(nrand.normal(size=(n,n)))
    omega = omega/la.norm(omega)
    return (sla.expm(delta*omega).dot(Y0) for delta in deltas)

def rand_pt_ball(r, c):
    n = len(c)
    def rand_box():
        return (2*nrand.random((n, 1)) - 1)*r
    pt = rand_box()
    while la.norm(pt) > r:
        pt = rand_box()
    c = c.reshape([n, 1])
    return pt + c

def stoch_ball(rs, cs, cts):
    """
    Generate points from balls with radii rs, and centers cs
    Number of pts from each ball is given by cts
    """
    balls = []
    for (r, c, ct) in zip(rs, cs, cts):
        guy = [rand_pt_ball(r, c) for j in range(ct)]
        ball = np.hstack(guy)
        balls.append(ball)
    return np.hstack(balls)
    

def dist_mat(pts):
    n,c = pts.shape
    return np.array([[la.norm(pts[:,i]-pts[:,j]) for i in range(c)] for j in range(c)])

def stoch_ball_basis(rs, cs, cts):
    nballs = len(rs)
    pts = stoch_ball(rs, cs, cts)
    D = dist_mat(pts)
    return spla.eigsh(D, k=nballs)[1]

def stoch_ball_D(rs, cs, cts):
    nballs = len(rs)
    pts = stoch_ball(rs, cs, cts)
    D = dist_mat(pts)
    return D

def stoch_ball_nup(rs, cs, cts):
    nballs = len(rs)
    pts = stoch_ball(rs, cs, cts)
    return NuP(pts)

def simp_stoch_ball_basis(nballs, npts, dist_cent):
    """
    Make basis from D matrix of points drawn from "nballs" balls
    arranged in a simplex
    with distance between centers given by dist_cent.
    and radii 1
    npts are drawn from each ball.
    """
    cs = np.eye(nballs)*dist_cent/np.sqrt(2) #Centers on simplex with distance dist_cent from eachother.
    return stoch_ball_basis([1]*nballs, [cs[:,i] for i in range(nballs)], [npts]*nballs) 


    
class NuP:
    def __init__(self, P):
        """P is a d x n matrix where each column is thought of as a point in RR^n."""
        self.nu = np.sum(P**2, axis=0)[:,None]
        self.P = P
    
def simp_stoch_ball_D(nballs, npts, dist_cent):
    """
    Make basis from D matrix of points drawn from "nballs" balls
    arranged in a simplex
    with distance between centers given by dist_cent.
    and radii 1
    npts are drawn from each ball.
    """
    cs = np.eye(nballs)*dist_cent/np.sqrt(2) #Centers on simplex with distance dist_cent from eachother.
    return stoch_ball_D([1]*nballs, [cs[:,i] for i in range(nballs)], [npts]*nballs) 

def simp_stoch_ball_nup(nballs, npts, dist_cent):
    cs = np.eye(nballs)*dist_cent/np.sqrt(2) #Centers on simplex with distance dist_cent from eachother.
    return stoch_ball_nup([1]*nballs, [cs[:,i] for i in range(nballs)], [npts]*nballs)

                    
