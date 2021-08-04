import my_test_YY
import numpy as np
import NMI_scores
from my_Cluster_Pblm import Cluster_Pblm

''' Soybean Data'''
"soybean-small.data"

'''Postal Codes Data'''

'''Spam Email Database'''
"spambase.data"


infile = "soybean-small.data"

with open(infile, "r") as data:
   data_matrix = np.asarray([[float(num) for num in line.split(',')] for line in data])
print(data_matrix)

'''Test on Shuffled Data'''
#np.random.shuffle(data_matrix)


#data_matrix = np.array([[1, 2], [2, 4]])

pblm = Cluster_Pblm(np.transpose(data_matrix), 4)
data.close()


# Returns list of dictionaries
clusters = my_test_YY.find_solution_matrix(pblm)

NMI_scores.find_NMI(clusters, [10, 10, 10, 17])

exit()