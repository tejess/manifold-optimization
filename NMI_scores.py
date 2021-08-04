from sklearn.metrics.cluster import normalized_mutual_info_score

def find_NMI(clusters, pts_per_cluster):
    # Find the known solution
    solution = [[i + 1 for i in range(pts_per_cluster[0])]]
    for i in range(1, len(pts_per_cluster)):
        cluster = [j + 1 for j in range(sum(pts_per_cluster[:i]), sum(pts_per_cluster[:i]) + pts_per_cluster[i])]
        solution.append(cluster)

    # Find NMI Scores
    sol_labels = find_labels(solution, pts_per_cluster)

    prediction = list(clusters.values())
    pred_labels = find_labels(prediction, pts_per_cluster)
    NMI = normalized_mutual_info_score(sol_labels, pred_labels)
    print(NMI)

def find_labels(A, pts_per_cluster):
    labels = [i for i in range(1, sum(pts_per_cluster) + 1)]
    for i in range(len(A)):
        for j in A[i]:
            labels[j - 1] = i + 1
    return labels

