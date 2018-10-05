# Made To Operate On Two Sets of Vectors
import numpy as np

from knn.non_broadcast_distance_metrics import manhattan_dist, hamming_dist, chisqr_dist


# Euclidean Distance aka L2-Norm
def euclidean(vectors_a, vectors_b):
    return np.sqrt(np.sum(np.square(vectors_b), axis=1)[:, np.newaxis] + np.sum(np.square(vectors_a), axis=1) -
                   2*np.dot(vectors_b, vectors_a.T))


# Manhattan Distance aka L1-Norm
# Uses Cython Function To Avoid High Memory Usage From Broadcasting
def manhattan(vectors_a, vectors_b):
    return manhattan_dist(vectors_a, vectors_b)


# Hamming Distance
# Uses Cython Function To Avoid High Memory Usage From Broadcasting
def hamming(vectors_a, vectors_b):
    return hamming_dist(vectors_a, vectors_b)


# Cosine Distance aka 1-Cosine Similarity
def cosine(vectors_a, vectors_b):
    norms_a = np.sqrt(np.sum(np.square(vectors_a), axis=1))
    norms_b = np.sqrt(np.sum(np.square(vectors_b), axis=1))
    norms_a_cross_norms_b = np.outer(norms_a, norms_b)
    sim = np.divide(np.dot(vectors_a, vectors_b.T), norms_a_cross_norms_b,
                    out=np.full([vectors_a.shape[0], vectors_b.shape[0]], np.nan), where=(norms_a_cross_norms_b != 0))
    return 1-sim.T


# Pearson Distance aka 1-Correlation
def pearson(vectors_a, vectors_b):
    mean_removed_a = (vectors_a-np.mean(vectors_a, axis=1)[:, np.newaxis])
    mean_removed_b = (vectors_b-np.mean(vectors_b, axis=1)[:, np.newaxis])
    std_dev_train = np.sqrt(np.sum(np.square(mean_removed_a), axis=1))
    std_dev_test = np.sqrt(np.sum(np.square(mean_removed_b), axis=1))
    std_dev_crossed = np.outer(std_dev_train, std_dev_test)
    correlation = np.divide(np.dot(mean_removed_a, mean_removed_b.T), std_dev_crossed,
                            out=np.zeros((vectors_a.shape[0], vectors_b.shape[0])), where=(std_dev_crossed != 0))
    return 1-correlation.T


# Chi-Square Statistic - Treat The Two Vectors As A Two Way Contingency Table
# Uses Cython Function To Avoid High Memory Usage From Broadcasting
def chisqr(vectors_a, vectors_b):
    return chisqr_dist(vectors_a, vectors_b)
