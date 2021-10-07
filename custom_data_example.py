import my_test_YY
import numpy as np
import NMI_scores
from my_Cluster_Pblm import Cluster_Pblm
import random

''' Soybean Data'''
"soybean-small.data"
# [10, 10, 10, 17]

'''Wine Data'''
"wine.data"
# [59, 71, 48]


infile = "wine.data"

with open(infile, "r") as data:
    data_matrix = np.asarray([[float(num) for num in line.split(',')] for line in data])
print(data_matrix)

'''Test on Shuffled Data'''
# np.random.shuffle(data_matrix)


pblm = Cluster_Pblm(np.transpose(data_matrix), 3)
data.close()

#Returns list of dictionaries
random.seed(6)
gamma = pow(10, -15)
eta = pow(10, -10)
rho = pow(10, 7)
clusters,_ = my_test_YY.find_solution_matrix(pblm, f'Gamma: {gamma}, Eta: {eta}, Rho: {rho}', gamma, eta, rho)

NMI_scores.find_NMI(clusters, [59, 71, 48])


'''My Function - Generates Y*Y^t matrix

iter = 1
best_par = []
for i in range(8, 12):
    for j in range(8, 12):
        for k in range(7, 10):
            for n in range(1, 2):
                gamma = pow(10, -i)
                eta = pow(10, -j)
                rho = n*pow(10, k)
                random.seed(6)
                clusterings, _ = my_test_YY.find_solution_matrix(pblm, iter, gamma, eta, rho)
                NMI = NMI_scores.find_NMI(clusterings, [59, 71, 48])
                if NMI >= 0.3:
                    best_par.append([gamma, eta, rho])
                iter += 1

print("Filtering Parameters")
sol_matrices = []
for sol in best_par:
    for i in range(3):
        random.seed(i+4)
        clusterings, Ys = my_test_YY.find_solution_matrix(pblm, i + 1, sol[0], sol[1], sol[2])
        NMI = NMI_scores.find_NMI(clusterings, [59, 71, 48])
        if NMI < 0.3:
            break
        if i == 2:
            sol_matrices.append([sol, clusterings, Ys])

print("Printing Figures")

for sol, clusterings, Ys in sol_matrices:
    my_test_YY.find_block_Y_ans(Ys, clusterings, sol)
'''