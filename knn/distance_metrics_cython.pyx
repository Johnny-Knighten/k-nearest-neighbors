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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline double euclidean(double[::1] vector1, double[::1] vector2):
    cdef double distance = 0.0
    cdef int dims = vector1.shape[0]
    cdef double temp
    cdef size_t i

    for i in range(0, dims):
        temp = vector1[i] - vector2[i]
        distance += (temp*temp)

    return sqrt(distance)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline double[:, ::1] _euclidean_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline double manhattan(double[::1] vector1, double[::1] vector2):
    cdef double distance = 0.0
    cdef int dims = vector1.shape[0]
    cdef double temp
    cdef size_t i

    for i in range(0, dims):
        temp = abs(vector1[i] - vector2[i])
        distance += temp

    return distance

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline double[:, ::1] _manhattan_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline double hamming(double[::1] vector1, double[::1] vector2):
    cdef double distance = 0.0
    cdef int dims = vector1.shape[0]
    cdef size_t i

    for i in range(0, dims):
        if vector1[i] != vector2[i]:
            distance += 1.0

    return distance

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline double[:, ::1] _hamming_pairwise(double[:, ::1] vectors_a, double[:, ::1] vectors_b):
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
