# Made To Operate On Two Sets of Vectors
import numpy as np


# Euclidean Distance aka L2-Norm
def euclidean(vectors_a, vectors_b):
    return np.sqrt(np.sum(np.square(vectors_a-vectors_b[:, np.newaxis]), axis=2))


# Manhattan Distance aka L1-Norm
def manhattan(vectors_a, vectors_b):
    return np.sum(np.abs(vectors_a-vectors_b[:, np.newaxis]), axis=2)


# Hamming Distance
def hamming(vectors_a, vectors_b):
    return np.sum(np.abs(vectors_a != vectors_b[:, np.newaxis]), axis=2)


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
def chisqr(vectors_a, vectors_b):
    all_col_sum = vectors_a + vectors_b[:, np.newaxis]
    all_col_sum_recip = np.reciprocal(all_col_sum, where=(all_col_sum != 0.0))
    vector_train_sum = np.sum(vectors_a, axis=1)
    vector_test_sum = np.sum(vectors_b, axis=1)

    rel_freq_train = np.divide(vectors_a, vector_train_sum[:, np.newaxis],
                               out=np.full([vectors_a.shape[0], vectors_a.shape[1]], np.nan),
                               where=(vector_train_sum[:, np.newaxis] != 0))

    rel_freq_test = np.divide(vectors_b, vector_test_sum[:, np.newaxis],
                              out=np.full([vectors_b.shape[0], vectors_b.shape[1]], np.nan),
                              where=(vector_test_sum[:, np.newaxis] != 0))

    diff_rel_freq_squared = np.square(rel_freq_train-rel_freq_test[:, np.newaxis])
    chisqr = np.sqrt(np.sum(all_col_sum_recip * diff_rel_freq_squared, axis=2))
    return chisqr
