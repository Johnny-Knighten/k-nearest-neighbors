import cython
from libc.math cimport sqrt
import numpy as np
cimport numpy as np


######################
# Euclidean Distance #
######################

# Python Wrapper
def euclidean_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
    return np.asarray(_euclidean_pairwise(vectors_a, vectors_b))

def euclidean(double[::1] vectors_a, double[::1] vectors_b):
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

# Python Wrapper
def manhattan_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
    return np.asarray(_manhattan_pairwise(vectors_a, vectors_b))

# Python Wrapper
def manhattan(double[::1] vectors_a, double[::1] vectors_b):
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
                temp = abs(vectors_a[j, k] - vectors_b[i, k])
                distance += temp

            distances[i, j] = distance

    return distances


####################
# Hamming Distance #
####################

# Python Wrapper
def hamming_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
    return np.asarray(_hamming_pairwise(vectors_a, vectors_b))

def hamming(double[::1] vectors_a, double[::1] vectors_b):
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
                    distance += 1.0

            distances[i, j] = distance

    return distances


#######################
# Chi-Square Distance #
#######################

# Python Wrapper
def chisqr_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
    return np.asarray(_chisqr_pairwise(vectors_a, vectors_b))

def chisqr(double[::1] vectors_a, double[::1] vectors_b):
    return _chisqr(vectors_a, vectors_b)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double _chisqr(double[::1] vectors_a, double[::1] vectors_b):
    cdef double distance = 0.0
    cdef int dims = vectors_a.shape[0]
    cdef int i
    cdef double sum_a = 0.0
    cdef double sum_b = 0.0

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef _chisqr_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):

    cdef int numb_vectors_a = vectors_a.shape[0]
    cdef int numb_vectors_b = vectors_b.shape[0]
    cdef int numb_dims = vectors_a.shape[1]
    cdef double[:, ::1] distance = np.zeros([numb_vectors_b, numb_vectors_a], dtype=np.float)

    cdef int i, j, k

    cdef double[::1] sum_a = np.zeros([numb_vectors_a], dtype=np.float)
    cdef double[::1] sum_b = np.zeros([numb_vectors_b], dtype=np.float)

    for i in range(numb_vectors_a):
        for j in range(numb_dims):
            sum_a[i] += vectors_a[i, j]

    for i in range(numb_vectors_b):
        for j in range(numb_dims):
            sum_b[i] += vectors_b[i, j]

    cdef double score
    cdef double col_sum

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
