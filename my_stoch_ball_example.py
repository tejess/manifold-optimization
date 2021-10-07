import my_test_YY
import NMI_scores
import random


'''Simulation parameters from the paper'''
separation = 2.05
nballs = 4
pts_per_ball = [22, 18, 19, 21]

# get a random Cluster_Pblm object.
pblm = my_test_YY.stoch_ball_pblm_uneven(pts_per_ball, separation)

random.seed(6)
gamma = pow(10, -15)
eta = pow(10, -14)
rho = pow(10, 7)
clusterings,_= my_test_YY.find_solution_matrix(pblm, f'Gamma: {gamma}, Eta: {eta}, Rho: {rho}', gamma, eta, rho)
NMI = NMI_scores.find_NMI(clusterings, pts_per_ball)

'''My Function - Generates Y*Y^t matrix
iter = 1
best_par = []
for i in range(1, 9):
    for j in range(1, 9):
        for n in range(1, 2):
            for k in range(1, 9):
                random.seed(6)
                gamma = pow(10, -i)
                eta = pow(10, -j)
                rho = n*pow(10, k)
                clusterings,_= my_test_YY.find_solution_matrix(pblm, iter, gamma, eta, rho)
                NMI = NMI_scores.find_NMI(clusterings, pts_per_ball)
                if NMI >= 1:
                    best_par.append([gamma, eta, rho])
                iter += 1

print("Filtering Parameters")

sol_matrices = []
for sol in best_par:
    for i in range(3):
        random.seed(i+4)
        clusterings, Ys = my_test_YY.find_solution_matrix(pblm, i+1, sol[0], sol[1], sol[2])
        NMI = NMI_scores.find_NMI(clusterings, pts_per_ball)
        if NMI < 1:
            break
        if i == 2:
            sol_matrices.append([sol, Ys])

print("Printing Figures")

for sol, Ys in sol_matrices:
    my_test_YY.find_pure_Y_ans(Ys, f'Gamma: {sol[0]}, Eta: {sol[1]}, Rho: {sol[2]}')
'''