
# Made To Operate On Two Vectors
import numpy as np


# Euclidean Distance aka L2-Norm
def euclidean(vector_1, vector_2):
    difference = vector_1-vector_2
    distance = np.sqrt(np.dot(difference, difference))
    return distance


# Manhattan Distance aka L1-Norm
def manhattan(vector_1, vector_2):
    difference = vector_1-vector_2
    absolute_diff = np.abs(difference)
    distance = np.sum(absolute_diff)
    return distance


# Hamming Distance
def hamming(vector_1, vector_2):
    return np.sum(vector_1 != vector_2)


# Cosine Distance aka 1-Cosine Similarity
def cosine(vector_1, vector_2):
    dot_prob = np.dot(vector_1, vector_2)
    vector_1_norm = np.sqrt(np.dot(vector_1, vector_1))
    vector_2_norm = np.sqrt(np.dot(vector_2, vector_2))

    # Cosine Similarity Is Undefined When A Vector Is All Zeros
    if vector_1_norm == 0 or vector_2_norm == 0:
        return np.nan

    cosine_sim = dot_prob/(vector_1_norm*vector_2_norm)
    return 1-cosine_sim


# Pearson Distance Between Two Vectors
def pearson(vector_1, vector_2):
    vector_1_centered = vector_1-np.mean(vector_1)
    vector_2_centered = vector_2-np.mean(vector_2)
    vector_1_stddev = np.sqrt(np.dot(vector_1_centered, vector_1_centered))
    vector_2_stddev = np.sqrt(np.dot(vector_2_centered, vector_2_centered))
    covariance = np.dot(vector_1_centered, vector_2_centered)

    # 0 Correlation If A Standard Deviation Is 0
    if vector_1_stddev == 0 or vector_2_stddev == 0:
        return 1

    correlation = covariance/(vector_1_stddev*vector_2_stddev)
    return 1-correlation


# Treat The Two Vectors As A Two Way Contingency Table
# Suitable For Non-Negative Data Such As Histograms
# See: "A Recent Advance in Data Analysis: Clustering Objects into Classes Characterized by Conjunctive Concepts"
#       Michalski, R. S. et al.
#       http://mars.gmu.edu/jspui/handle/1920/1556?show=full
# See: https://stats.stackexchange.com/questions/184101/comparing-two-histograms-using-chi-square-distance
def chisqr(vector_1, vector_2):
    col_sums = vector_1 + vector_2
    col_sums_recip = np.reciprocal(col_sums, where=(col_sums != 0.0))
    vector_1_sum = np.sum(vector_1)
    vector_2_sum = np.sum(vector_2)

    if vector_1_sum == 0. or vector_2_sum == 0.:
        return np.nan

    rel_freq_vector_1 = vector_1/vector_1_sum
    rel_freq_vector_2 = vector_2/vector_2_sum
    diff_rel_freq = rel_freq_vector_1-rel_freq_vector_2
    diff_rel_freq_square = np.square(diff_rel_freq)
    chisqr = np.sqrt(np.dot(col_sums_recip, diff_rel_freq_square))
    return chisqr
