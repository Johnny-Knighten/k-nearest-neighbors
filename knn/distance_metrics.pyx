import cython
from libc.math cimport sqrt
import numpy as np
cimport numpy as np


######################
# Euclidean Distance #
######################

def euclidean(double[::1] vectors_a, double[::1] vectors_b):
    """ Finds the euclidean distance between two vectors.

    This is a python wrapper for the cython implementation.

    Args:
        vectors_a (ndarray): the first 1D array must be of type np.float
        vectors_b (ndarray): the second 1D array must be of type np.float

    Returns:
        (np.float): the euclidean distance between the two supplied vectors.

    """
    return _euclidean(vectors_a, vectors_b)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double _euclidean(double[::1] vector1, double[::1] vector2):
    cdef double distance = 0.0
    cdef int dims = vector1.shape[0]
    cdef double temp
    cdef int i

    for i in range(0, dims):
        temp = vector1[i] - vector2[i]
        distance += (temp*temp)

    return sqrt(distance)

def euclidean_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
    """ Finds the euclidean distance between all pairs of vectors in the two supplied matrices.

    This is a python wrapper for the cython implementation.

    Args:
        vectors_a (ndarray): the first 2D array of vectors must be of type np.float
        vectors_b (ndarray): the second 2D array of vectors must be of type np.float

    Returns:
        (ndarray): A 2D array containing the euclidean distances between the vectors in the matrices provided. Rows
            correspond to the vectors in vectors_b and the columns correspond to vectors_a.

    """
    return np.asarray(_euclidean_pairwise(vectors_a, vectors_b))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double[:, ::1] _euclidean_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
    cdef int numb_vectors_a = vectors_a.shape[0]
    cdef int numb_vectors_b = vectors_b.shape[0]
    cdef int numb_dims = vectors_a.shape[1]
    cdef double[:, ::1] distances = np.zeros([numb_vectors_b, numb_vectors_a])

    cdef int i, j, k
    cdef double distance, temp

    for i in range(numb_vectors_b):
        for j in range(numb_vectors_a):
            distance = 0.0
            for k in range(numb_dims):
                temp = vectors_a[j, k] - vectors_b[i, k]
                distance += (temp*temp)

            distances[i, j] = sqrt(distance)

    return distances


######################
# Manhattan Distance #
######################

def manhattan(double[::1] vectors_a, double[::1] vectors_b):
    """ Finds the manhattan distance between two vectors.

    This is a python wrapper for the cython implementation.

    Args:
        vectors_a (ndarray): the first 1D array must be of type np.float
        vectors_b (ndarray): the second 1D array must be of type np.float

    Returns:
        (np.float): the manhattan distance between the two supplied vectors.

    """
    return _manhattan(vectors_a, vectors_b)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double _manhattan(double[::1] vector1, double[::1] vector2):
    cdef double distance = 0.0
    cdef int dims = vector1.shape[0]

    cdef double temp
    cdef int i

    for i in range(0, dims):
        temp = abs(vector1[i] - vector2[i])
        distance += temp

    return distance

def manhattan_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
    """ Finds the manhattan distance between all pairs of vectors in the two supplied matrices.

    This is a python wrapper for the cython implementation.

    Args:
        vectors_a (ndarray): the first 2D array of vectors must be of type np.float
        vectors_b (ndarray): the second 2D array of vectors must be of type np.float

    Returns:
        (ndarray): A 2D array containing the manhattan distances between the vectors in the matrices provided. Rows
            correspond to the vectors in vectors_b and the columns correspond to vectors_a.

    """
    return np.asarray(_manhattan_pairwise(vectors_a, vectors_b))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double[:, ::1] _manhattan_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
    cdef int numb_vectors_a = vectors_a.shape[0]
    cdef int numb_vectors_b = vectors_b.shape[0]
    cdef int numb_dims = vectors_a.shape[1]
    cdef double[:, ::1] distances = np.zeros([numb_vectors_b, numb_vectors_a])

    cdef int i, j, k
    cdef double distance
    cdef double temp

    for i in range(numb_vectors_b):
        for j in range(numb_vectors_a):
            distance = 0.0
            for k in range(numb_dims):
                distances[i, j] += abs(vectors_a[j, k] - vectors_b[i, k])

    return distances


####################
# Hamming Distance #
####################

def hamming(double[::1] vectors_a, double[::1] vectors_b):
    """ Finds the hamming distance between two vectors.

    This is a python wrapper for the cython implementation.

    Args:
        vectors_a (ndarray): the first 1D array must be of type np.float
        vectors_b (ndarray): the second 1D array must be of type np.float

    Returns:
        (np.float): the hamming distance between the two supplied vectors.

    """
    return _hamming(vectors_a, vectors_b)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double _hamming(double[::1] vector1, double[::1] vector2):
    cdef double distance = 0.0
    cdef int dims = vector1.shape[0]

    cdef int i

    for i in range(0, dims):
        if vector1[i] != vector2[i]:
            distance += 1.0

    return distance

def hamming_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
    """ Finds the hamming distance between all pairs of vectors in the two supplied matrices.

    This is a python wrapper for the cython implementation.

    Args:
        vectors_a (ndarray): the first 2D array of vectors must be of type np.float
        vectors_b (ndarray): the second 2D array of vectors must be of type np.float

    Returns:
        (ndarray): A 2D array containing the hamming distances between the vectors in the matrices provided. Rows
            correspond to the vectors in vectors_b and the columns correspond to vectors_a.

    """
    return np.asarray(_hamming_pairwise(vectors_a, vectors_b))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double[:, ::1] _hamming_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
    cdef int numb_vectors_a = vectors_a.shape[0]
    cdef int numb_vectors_b = vectors_b.shape[0]
    cdef int numb_dims = vectors_a.shape[1]
    cdef double[:, ::1] distances = np.zeros([numb_vectors_b, numb_vectors_a])

    cdef int i, j, k
    cdef double distance
    cdef double temp

    for i in range(numb_vectors_b):
        for j in range(numb_vectors_a):
            distance = 0.0
            for k in range(numb_dims):
                if vectors_a[j, k] != vectors_b[i, k]:
                    distances[i, j] += 1.0

    return distances


#######################
# Chi-Square Distance #
#######################

def chisqr(double[::1] vectors_a, double[::1] vectors_b):
    """ Finds the chi-squared distance between two vectors.

    This is a python wrapper for the cython implementation.

    Args:
        vectors_a (ndarray): the first 1D array must be of type np.float
        vectors_b (ndarray): the second 1D array must be of type np.float

    Returns:
        (np.float): the chi-squared distance between the two supplied vectors.

    """
    return _chisqr(vectors_a, vectors_b)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double _chisqr(double[::1] vectors_a, double[::1] vectors_b):
    cdef double distance = 0.0
    cdef int dims = vectors_a.shape[0]
    cdef double sum_a = 0.0
    cdef double sum_b = 0.0

    cdef int i

    for i in range(dims):
        sum_a += vectors_a[i]
        sum_b += vectors_b[i]

    if sum_a == 0.0 or sum_b == 0.0:
        return np.nan

    for i in range(dims):
        col_sum = vectors_a[i] + vectors_b[i]
        if col_sum == 0:
            continue

        distance += (1.0/col_sum) * ((vectors_a[i]/sum_a)-(vectors_b[i]/sum_b))**2

    return sqrt(distance)

def chisqr_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
    """ Finds the chi-squared distance between all pairs of vectors in the two supplied matrices.

    This is a python wrapper for the cython implementation.

    Args:
        vectors_a (ndarray): the first 2D array of vectors must be of type np.float
        vectors_b (ndarray): the second 2D array of vectors must be of type np.float

    Returns:
        (ndarray): A 2D array containing the chi-squared distances between the vectors in the matrices provided. Rows
            correspond to the vectors in vectors_b and the columns correspond to vectors_a.

    """
    return np.asarray(_chisqr_pairwise(vectors_a, vectors_b))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double[:, ::1] _chisqr_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
    cdef int numb_vectors_a = vectors_a.shape[0]
    cdef int numb_vectors_b = vectors_b.shape[0]
    cdef int numb_dims = vectors_a.shape[1]
    cdef double[:, ::1] distance = np.zeros([numb_vectors_b, numb_vectors_a], dtype=np.float)
    cdef double[::1] sum_a = np.zeros([numb_vectors_a], dtype=np.float)
    cdef double[::1] sum_b = np.zeros([numb_vectors_b], dtype=np.float)

    cdef int i, j, k
    cdef double score
    cdef double col_sum

    for i in range(numb_vectors_a):
        for j in range(numb_dims):
            sum_a[i] += vectors_a[i, j]

    for i in range(numb_vectors_b):
        for j in range(numb_dims):
            sum_b[i] += vectors_b[i, j]

    for i in range(numb_vectors_b):
        for j in range(numb_vectors_a):
            score = 0
            for k in range(numb_dims):
                col_sum = vectors_a[j, k] + vectors_b[i, k]
                if col_sum == 0:
                    continue

                score += (1.0/col_sum) * ((vectors_a[j, k]/sum_a[j])-(vectors_b[i, k]/sum_b[i]))**2
            distance[i, j] = score**0.5

    return distance


###################
# Cosine Distance #
###################

def cosine(double[::1] vectors_a, double[::1] vectors_b):
    """ Finds the cosine distance between two vectors.

    This is a python wrapper for the cython implementation.

    Args:
        vectors_a (ndarray): the first 1D array must be of type np.float
        vectors_b (ndarray): the second 1D array must be of type np.float

    Returns:
        (np.float): the cosine distance between the two supplied vectors.

    """
    return _cosine(vectors_a, vectors_b)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double _cosine(double[::1] vectors_a, double[::1] vectors_b):
    cdef double distance = 0.0
    cdef int dims = vectors_a.shape[0]
    cdef double vecta_norm = 0.0
    cdef double vectb_norm = 0.0
    cdef double dot_prod = 0.0

    cdef int i

    for i in range(dims):
        dot_prod += (vectors_a[i]*vectors_b[i])
        vecta_norm += (vectors_a[i]*vectors_a[i])
        vectb_norm += (vectors_b[i]*vectors_b[i])

    vecta_norm = sqrt(vecta_norm)
    vectb_norm = sqrt(vectb_norm)

    # Cosine Similarity Is Undefined When A Vector Is All Zeros
    if vecta_norm == 0 or vectb_norm == 0:
        return np.nan

    cos_sim = dot_prod / (vecta_norm * vectb_norm)

    return 1-cos_sim

def cosine_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
    """ Finds the cosine distance between all pairs of vectors in the two supplied matrices.

    This is a python wrapper for the cython implementation.

    Args:
        vectors_a (ndarray): the first 2D array of vectors must be of type np.float
        vectors_b (ndarray): the second 2D array of vectors must be of type np.float

    Returns:
        (ndarray): A 2D array containing the cosine distances between the vectors in the matrices provided. Rows
            correspond to the vectors in vectors_b and the columns correspond to vectors_a.

    """
    return np.asarray(_cosine_pairwise(vectors_a, vectors_b))

cdef double[:, ::1] _cosine_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
    cdef int numb_vectors_a = vectors_a.shape[0]
    cdef int numb_vectors_b = vectors_b.shape[0]
    cdef int dims = vectors_a.shape[1]
    cdef double[:, ::1] distance = np.zeros([numb_vectors_b, numb_vectors_a], dtype=np.float)
    cdef double[::1] vecta_norms = np.zeros(numb_vectors_a)
    cdef double[::1] vectb_norms = np.zeros(numb_vectors_b)
    cdef double[: ,::1] dot_prods = np.zeros((numb_vectors_b, numb_vectors_a))

    cdef int i, j, k

    for i in range(numb_vectors_a):
        for j in range(dims):
            vecta_norms[i] += (vectors_a[i, j] * vectors_a[i, j])
        vecta_norms[i] = sqrt(vecta_norms[i])

    for i in range(numb_vectors_b):
        for j in range(dims):
            vectb_norms[i] += (vectors_b[i, j] * vectors_b[i, j])
        vectb_norms[i] = sqrt(vectb_norms[i])

    for i in range(numb_vectors_b):
        for j in range(numb_vectors_a):
            for k in range(dims):
                dot_prods[i, j] += (vectors_b[i, k] * vectors_a[j, k])

    for i in range(numb_vectors_b):
        for j in range(numb_vectors_a):
            if vecta_norms[j] == 0.0 or vectb_norms[i] == 0.0:
                distance[i, j] = np.nan
                continue

            distance[i, j] = 1 - (dot_prods[i,j] / (vecta_norms[j] * vectb_norms[i]))

    return distance


####################
# Pearson Distance #
####################

def pearson(double[::1] vectors_a, double[::1] vectors_b):
    """ Finds the pearson distance between two vectors.

    This is a python wrapper for the cython implementation.

    Args:
        vectors_a (ndarray): the first 1D array must be of type np.float
        vectors_b (ndarray): the second 1D array must be of type np.float

    Returns:
        (np.float): the pearson distance between the two supplied vectors.

    """
    return _pearson(vectors_a, vectors_b)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double _pearson(double[::1] vectors_a, double[::1] vectors_b):
    cdef int dims = vectors_a.shape[0]
    cdef double mean_a = 0.0
    cdef double mean_b = 0.0
    cdef double vect_a_stddev = 0.0
    cdef double vect_b_stddev = 0.0
    cdef double covariance = 0.0

    cdef int i

    for i in range(dims):
        mean_a += vectors_a[i]
        mean_b += vectors_b[i]

    mean_a /= dims
    mean_b /= dims

    for i in range(dims):
        vect_a_stddev += (vectors_a[i] - mean_a) ** 2
        vect_b_stddev += (vectors_b[i] - mean_b) ** 2
        covariance += ((vectors_a[i] - mean_a) * (vectors_b[i] - mean_b))

    vect_a_stddev = sqrt(vect_a_stddev)
    vect_b_stddev = sqrt(vect_b_stddev)

    if vect_a_stddev == 0.0 or vect_b_stddev == 0.0:
        return 1.0

    correlation = covariance/(vect_a_stddev*vect_b_stddev)

    return 1-correlation

def pearson_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
    """ Finds the pearson distance between all pairs of vectors in the two supplied matrices.

    This is a python wrapper for the cython implementation.

    Args:
        vectors_a (ndarray): the first 2D array of vectors must be of type np.float
        vectors_b (ndarray): the second 2D array of vectors must be of type np.float

    Returns:
        (ndarray): A 2D array containing the pearson distances between the vectors in the matrices provided. Rows
            correspond to the vectors in vectors_b and the columns correspond to vectors_a.

    """
    return np.asarray(_pearson_pairwise(vectors_a, vectors_b))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double[:, ::1] _pearson_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
    cdef int numb_vectors_a = vectors_a.shape[0]
    cdef int numb_vectors_b = vectors_b.shape[0]
    cdef int dims = vectors_a.shape[1]
    cdef double[::1] vect_a_means = np.zeros(numb_vectors_a)
    cdef double[::1] vect_b_means = np.zeros(numb_vectors_b)
    cdef double[::1] vect_a_stddevs = np.zeros(numb_vectors_a)
    cdef double[::1] vect_b_stddevs = np.zeros(numb_vectors_b)
    cdef double[:, ::1] covariances = np.zeros((numb_vectors_b, numb_vectors_a))
    cdef double[:, ::1] pearson_scores = np.zeros((numb_vectors_b, numb_vectors_a))

    cdef int i, j

    for i in range(numb_vectors_a):
        for j in range(dims):
            vect_a_means[i] += vectors_a[i, j]
        vect_a_means[i] /= dims

    for i in range(numb_vectors_b):
        for j in range(dims):
            vect_b_means[i] += vectors_b[i, j]
        vect_b_means[i] /= dims

    for i in range(numb_vectors_b):
        for j in range(dims):
            vect_b_stddevs[i] += (vectors_b[i, j] - vect_b_means[i]) ** 2
        vect_b_stddevs[i] = sqrt(vect_b_stddevs[i])

    for i in range(numb_vectors_a):
        for j in range(dims):
            vect_a_stddevs[i] += (vectors_a[i, j] - vect_a_means[i]) ** 2
        vect_a_stddevs[i] = sqrt(vect_a_stddevs[i])

    for i in range(numb_vectors_b):
        for j in range(numb_vectors_a):
            for k in range(dims):
                covariances[i, j] += ((vectors_b[i, k] - vect_b_means[i]) * (vectors_a[j, k] - vect_a_means[j]))

    for i in range(numb_vectors_b):
        for j in range(numb_vectors_a):
            if  vect_b_stddevs[i] == 0.0 or vect_a_stddevs[j] == 0.0:
                pearson_scores[i, j] = 1
                continue
            pearson_scores[i, j] = 1 - (covariances[i, j] / (vect_a_stddevs[j] * vect_b_stddevs[i]))

    return pearson_scores
