import cython
import numpy as np
cimport numpy as np

# Manhattan Distance(L1-Norm)
# Used To Prevent Memory Overhead That Occurs When Using Numpy Broadcasting
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def manhattan_dist(np.ndarray[double, ndim=2] vectors_a, np.ndarray[double, ndim=2] vectors_b):

    cdef int numb_vectors_a = vectors_a.shape[0]
    cdef int numb_vectors_b = vectors_b.shape[0]
    cdef int numb_dims = vectors_a.shape[1]
    cdef np.ndarray[double, ndim=2] distance = np.zeros([numb_vectors_b, numb_vectors_a], dtype=np.float)

    cdef int i, j, k
    cdef double score

    for i in range(numb_vectors_b):
        for j in range(numb_vectors_a):
            score = 0
            for k in range(numb_dims):
                score += abs(vectors_a[j,k] - vectors_b[i,k])

            distance[i,j] = score

    return distance

# Hamming Distance
# Used To Prevent Memory Overhead That Occurs When Using Numpy Broadcasting
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def hamming_dist(np.ndarray[double, ndim=2] vectors_a, np.ndarray[double, ndim=2] vectors_b):

    cdef int numb_vectors_a = vectors_a.shape[0]
    cdef int numb_vectors_b = vectors_b.shape[0]
    cdef int numb_dims = vectors_a.shape[1]
    cdef np.ndarray[int, ndim=2] distance = np.zeros([numb_vectors_b, numb_vectors_a], dtype=np.int32)

    cdef int i, j, k
    cdef int score

    for i in range(numb_vectors_b):
        for j in range(numb_vectors_a):
            score = 0
            for k in range(numb_dims):
                score += vectors_a[j,k] != vectors_b[i,k]

            distance[i,j] = score

    return distance


# Chi-Squared
# Used To Prevent Memory Overhead That Occurs When Using Numpy Broadcasting
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def chisqr_dist(np.ndarray[double, ndim=2] vectors_a, np.ndarray[double, ndim=2] vectors_b):

    cdef int numb_vectors_a = vectors_a.shape[0]
    cdef int numb_vectors_b = vectors_b.shape[0]
    cdef int numb_dims = vectors_a.shape[1]
    cdef np.ndarray[double, ndim=2] distance = np.zeros([numb_vectors_b, numb_vectors_a], dtype=np.float)

    cdef int i, j, k

    cdef np.ndarray[double, ndim=1] sum_a = np.zeros([numb_vectors_a], dtype=np.float)
    cdef np.ndarray[double, ndim=1] sum_b = np.zeros([numb_vectors_b], dtype=np.float)

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