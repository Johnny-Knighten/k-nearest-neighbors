cdef double euclidean(double[::1], double[::1])
cdef double[:, ::1] _euclidean_pairwise(double[:, ::1], double[:, ::1])

cdef double manhattan(double[::1], double[::1])
cdef double[:, ::1] _manhattan_pairwise(double[:, ::1], double[:, ::1])

cdef double hamming(double[::1] , double[::1])
cdef double[:, ::1] _hamming_pairwise(double[:, ::1], double[:, ::1])