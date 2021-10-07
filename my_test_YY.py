import matplotlib.pyplot as plt
from my_Cluster_Pblm import Cluster_Pblm

from random_creators import *

DATAFOLDER = "data/"


###
### Functions for generating random problems
###

def stoch_ball_pblm(nballs, npts, dist_cent):
    """
    Creates a basic test problem.
    k = nballs.
    There are k balls in \real^k with equal spacing.  From each ball we draw an equal number of points (npts).
    The balls have radius one, and lie on the simplex with edge length dist_cent.
    """
    cs = np.eye(nballs) * dist_cent / np.sqrt(2)  # Centers on simplex with distance dist_cent from eachother.
    P = stoch_ball([1] * nballs, cs, [npts] * nballs)
    return Cluster_Pblm(P, nballs)


def stoch_ball_pblm_uneven(npts_list, dist_cent):
    """
    Similar to stoch_ball_pblm,
    but is able to draw different numbers of points from each ball.
    npts_list is the list of number of points, the number of balls is the length of this list.
    dist_cent is the distance between the centers.
    """
    nballs = len(npts_list)
    cs = np.eye(nballs) * dist_cent / np.sqrt(2)  # Centers on simplex with distance dist_cent from eachother.
    P = stoch_ball([1] * nballs, cs, npts_list)
    return Cluster_Pblm(P, nballs)


'''My Functions'''

# Finds and plots solution matrix
def find_solution_matrix(pblm, i=1, gamma=pow(10, -2), eta=pow(10, -4), rho=6*pow(10, 2)):
    Ys = pblm.do_path(gamma, eta, rho)
    clusters = find_cluster_assignments(Ys, pblm.k)
    print(f'Figure {i}: gamma = {gamma}, eta = {eta}, rho = {rho}')

    '''Plot Block Structure'''
    find_block_Y_ans(Ys, clusters, i)

    '''Plot Pure X = YY^t matrix'''
    #find_pure_Y_ans(Ys, i)

    #clusterings.append(clusters)
    return clusters, Ys


def find_pure_Y_ans(Y_sol, i):
    Y_ans = find_YYt(Y_sol)
    # save_matrix(Y_ans, i, a, b)
    plot_matrix(Y_ans, i)


def find_block_Y_ans(Y_sol, clusters, i,):
    sorted_Y_sol = point_sort(Y_sol, clusters)
    block_structure = find_YYt(sorted_Y_sol)
    # save_matrix(block_structure, i, a, b)
    plot_matrix(block_structure, i)


# Sorts so points in same cluster are next to each other
def point_sort(Y_sol, clusters):
    Y_sol_sorted = []
    for i in range(len(clusters)):
        for j in clusters[f'Cluster {i + 1}']:
            Y_sol_sorted.append(Y_sol[j - 1])
    return np.array(Y_sol_sorted)


# Finds a dictionary of cluster assignments
def find_cluster_assignments(Y, k):
    dict_of_clusters = {}
    for i in range(k):
        dict_of_clusters[f'Cluster {i + 1}'] = []
    j = 1
    try:
        for array in Y:
            result = np.where(array == np.amax(array))
            dict_of_clusters[f'Cluster {result[0][0] + 1}'].append(j)
            j+= 1
        return dict_of_clusters
    except:
        print("Index Error")
        return dict_of_clusters


# Saves matrix to an outfile
def save_matrix(Y, i, a, b):
    with open('outfile.txt', 'a') as f:
        f.write(f'Figure {i + 1}: a = {a}, b = {b} \n')
        for line in Y:
            np.savetxt(f, line, fmt='%.5f')
        f.write("\n")
    f.close()


# Finds the matrix YY^t from Y
def find_YYt(Y):
    # replaces negative values with 0
    Y = np.where(Y < 0, 0, Y)
    # Finds Y_ans = Y*Y^t
    Y_t = np.transpose(Y)
    Y_ans = np.matrix(Y @ Y_t)
    # normalize Y_ans to [0,1]
    Y_ans = Y_ans / np.max(Y_ans)
    return Y_ans


# Plots the resulting matrix
def plot_matrix(Y_ans, i):
    plt.matshow(Y_ans)
    plt.colorbar()
    plt.title(f'{i}')
    plt.xlabel('Wine Data')
    plt.show()


'''End of My Functions'''