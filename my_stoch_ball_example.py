import my_test_YY
import NMI_scores


'''Simulation parameters from the paper'''
separation = 2.05
nballs = 4
pts_per_ball = [22, 18, 19, 21]

# get a random Cluster_Pblm object.
pblm = my_test_YY.stoch_ball_pblm_uneven(pts_per_ball, separation)


'''My Function - Generates Y*Y^t matrix'''
for i in range(10):
    for j in range(10):
        for k in range(10):
            clusterings = my_test_YY.find_solution_matrix(pblm, gamma=pow(10, -2), eta=pow(10, -4), rho=6*pow(10, 2))

NMI_scores.find_NMI(clusterings, pts_per_ball)


