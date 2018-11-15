import cython
import numpy as np
cimport numpy as np
import math
from knn.distance_metrics_cython cimport _euclidean, _euclidean_pairwise, _manhattan, _manhattan_pairwise, _hamming,\
    _hamming_pairwise

ctypedef double (*metric_func)(double[::1], double[::1])
ctypedef double[:, ::1] (*pairwise_metric_fun)(double[:, ::1], double[:, ::1])

cdef class BallTree:

    cdef double[:, ::1] data_view
    cdef long[::1] data_inds_view
    cdef np.ndarray data
    cdef np.ndarray data_inds

    cdef double[:,::1] query_data_view

    cdef np.ndarray node_data_inds
    cdef np.ndarray node_radius
    cdef np.ndarray node_is_leaf
    cdef np.ndarray node_center
    cdef long[:, ::1] node_data_inds_view
    cdef double[::1] node_radius_view
    cdef double[::1] node_is_leaf_view
    cdef double[:, ::1] node_center_view

    cdef int leaf_size
    cdef int node_count
    cdef int tree_height

    cdef public np.ndarray heap
    cdef double[:, ::1] heap_view
    cdef public np.ndarray heap_inds
    cdef long[:, ::1] heap_inds_view

    cdef metric_func metric
    cdef pairwise_metric_fun pair_metric


    def __init__(self, data, leaf_size, metric="euclidean"):

        # Data
        self.data = np.asarray(data, dtype=np.float, order='C')
        self.data_view = memoryview(self.data)
        self.data_inds = np.arange(data.shape[0], dtype=np.int)
        self.data_inds_view = memoryview(self.data_inds)

        # Tree Shape
        self.leaf_size = leaf_size
        leaf_count = self.data.shape[0] / leaf_size
        self.tree_height = math.ceil(np.log2(leaf_count)) + 1
        self.node_count = int(2 ** self.tree_height) - 1

        # Node Data
        self.node_data_inds = np.zeros((self.node_count, 2), dtype=np.int, order='C')
        self.node_radius = np.zeros(self.node_count, order='C')
        self.node_is_leaf = np.zeros(self.node_count, order='C')
        self.node_center = np.zeros((self.node_count, data.shape[1]), order='C')
        self.node_data_inds_view = memoryview(self.node_data_inds)
        self.node_radius_view = memoryview(self.node_radius)
        self.node_is_leaf_view = memoryview(self.node_is_leaf)
        self.node_center_view = memoryview(self.node_center)

        if metric == "manhattan":
            self.metric = _manhattan
            self.pair_metric = _manhattan_pairwise
        elif metric == "hamming":
            self.metric = _hamming
            self.pair_metric = _hamming_pairwise
        else:
            self.metric = _euclidean
            self.pair_metric = _euclidean_pairwise


    def build_tree(self):
        self._build(0, 0, self.data.shape[0]-1)


    def _build(self, long node_index, long node_data_start, long node_data_end):

        ##########################
        # Current Node Is A Leaf #
        ##########################
        if (node_data_end-node_data_start+1) <= self.leaf_size:

            self.node_center[node_index] = np.mean(self.data[self.data_inds[node_data_start:node_data_end+1]], axis=0)

            self.node_radius[node_index] = np.max(self.pair_metric(self.data[self.data_inds[node_data_start:node_data_end+1]],
                                                            self.node_center[node_index,  :][np.newaxis, :]))

            self.node_data_inds[node_index, 0] = node_data_start
            self.node_data_inds[node_index, 1] = node_data_end

            self.node_is_leaf[node_index] = True
            return None

        #################################
        # Current Node Is Internal Node #
        #################################

        # Select Random Point -  x0
        rand_index = np.random.choice(node_data_end-node_data_start+1, 1, replace=False)
        rand_point = self.data[self.data_inds[rand_index], :]

        # Find Point Farthest Away From x0 - x1
        distances = self.pair_metric(self.data[self.data_inds[node_data_start:node_data_end+1]], rand_point)
        ind_of_max_dist = np.argmax(distances)
        max_vector_1 = self.data[ind_of_max_dist]

        # Find Point Farthest Away From x1 - x2
        distances = self.pair_metric(self.data[self.data_inds[node_data_start:node_data_end+1]], max_vector_1[np.newaxis, :])
        ind_of_max_dist = np.argmax(distances)
        max_vector_2 = self.data[ind_of_max_dist]

        # Project Data On Vector Between x1 and x2
        proj_data = np.dot(self.data[self.data_inds[node_data_start:node_data_end+1]], max_vector_1-max_vector_2)

        # Find Median Of Projected Data
        median = np.partition(proj_data, proj_data.size//2)[proj_data.size//2]

        # Split Data Around Median Using Hoare Partitioning
        low = node_data_start
        high = node_data_end
        pivot = median
        self._hoare_partition(pivot, low, high, proj_data)

        # Create Circle
        center = np.mean(self.data[self.data_inds[node_data_start:node_data_end+1]], axis=0)
        radius = np.max(self.pair_metric(self.data[self.data_inds[node_data_start:node_data_end+1]], center[np.newaxis, :]))

        self.node_data_inds[node_index, 0] = node_data_start
        self.node_data_inds[node_index, 1] = node_data_end

        self.node_radius[node_index] = radius
        self.node_center[node_index] = center

        self.node_is_leaf[node_index] = False

        # Build Children Circles
        left_index = 2 * node_index + 1
        right_index = left_index + 1
        self._build(left_index, node_data_start,  node_data_start+ (proj_data.size//2)-1 )
        self._build(right_index, node_data_start+(proj_data.size//2),   node_data_end)


    cdef int _hoare_partition(self, pivot, low, high, projected_data):

        i = low - 1
        j = high + 1
        i2 = -1
        j2 = projected_data.shape[0]

        while True:

            # Scan From Left To Find Value Greater Than Pivot
            condition = True
            while condition:
                i += 1
                i2 += 1
                condition = projected_data[i2] < pivot

            # Scan From Right To Find Value Less Than Pivot
            condition = True
            while condition:
                j -= 1
                j2 -= 1
                condition = projected_data[j2] > pivot

            # Time To End Algorithm
            if (i >= j):
                return j

            # Swap Values
            projected_data[i2], projected_data[j2] = projected_data[j2], projected_data[i2]
            self.data_inds[i], self.data_inds[j] = self.data_inds[j], self.data_inds[i]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def query(self, query_data, k):

        cdef int i
        cdef double[::1] query_vector, initial_center
        cdef int numb_query_vectors = query_data.shape[0]
        cdef double dist

        self.heap = np.full((query_data.shape[0], k), np.inf, order='C')
        self.heap_view = memoryview(self.heap)
        self.heap_inds = np.zeros((query_data.shape[0], k), dtype=np.int, order='C')
        self.heap_inds_view = memoryview(self.heap_inds)

        self.query_data_view = memoryview(query_data)

        initial_center = self.node_center_view[0]
        for i in range(0, numb_query_vectors):
            query_vector = self.query_data_view[i]
            dist = self.metric(initial_center, query_vector)
            self._query(i, dist, 0, query_vector)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef int _query(self, int query_vect_ind, double dist_to_cent, int curr_node, double[::1] query_data):

        cdef int i, child1, child2, lower_index, upper_index, curr_index
        cdef double child1_dist, child2_dist, dist
        cdef double[::1] curr_vect, child1_center, child2_center

        # Prune This Ball
        if dist_to_cent - self.node_radius_view[curr_node] > self._heap_peek_head(query_vect_ind):
            return 0

        # Currently A Leaf Node
        if self.node_is_leaf_view[curr_node]:
            lower_index = self.node_data_inds_view[curr_node][0]
            upper_index = self.node_data_inds_view[curr_node][1] + 1
            for i in range(lower_index, upper_index):
                curr_index = self.data_inds_view[i]
                curr_vect = self.data_view[curr_index]
                dist = self.metric(curr_vect, query_data)
                if dist < self._heap_peek_head(query_vect_ind):
                    self._heap_pop_push(query_vect_ind, dist, self.data_inds_view[i])

        # Not Leaf So Explore Children
        else:
            child1 = 2 * curr_node + 1
            child2 = child1 + 1

            child1_center = self.node_center_view[child1]
            child2_center = self.node_center_view[child2]

            child1_dist = self.metric(child1_center, query_data)
            child2_dist = self.metric(child2_center, query_data)

            if child1_dist <= child2_dist:
                self._query(query_vect_ind, child1_dist, child1, query_data)
                self._query(query_vect_ind, child2_dist, child2, query_data)
            else:
                self._query(query_vect_ind, child2_dist, child2, query_data)
                self._query(query_vect_ind, child1_dist, child1, query_data)

        return 0


    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cdef inline double _heap_peek_head(self, int level):
        return self.heap_view[level, 0]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef int _heap_pop_push(self, int level, double value, int index):

        cdef int left_ind, right_ind
        cdef int i
        cdef double temp_value
        cdef int temp_index

        # Put New Value At Head And Remove Old Value
        self.heap_view[level, 0] = value
        self.heap_inds_view[level, 0] = index

        # Update Heap Structure
        i = 0
        while True:
            left_ind = 2 * i + 1
            right_ind = left_ind + 1

            # Catch Edge Of Array
            if left_ind >= self.heap.shape[1]:
                break
            elif right_ind >= self.heap.shape[1]:
                if self.heap_view[level, left_ind] > self.heap_view[level, i]:
                    temp_value = self.heap_view[level, i]
                    self.heap_view[level, i] = self.heap_view[level, left_ind]
                    self.heap_view[level, left_ind] = temp_value

                    temp_index = self.heap_inds_view[level, i]
                    self.heap_inds_view[level, i] = self.heap_inds_view[level, left_ind]
                    self.heap_inds_view[level, left_ind] = temp_index
                break

            # Determine If We Should Explore Left or Right
            if self.heap_view[level, left_ind] > self.heap_view[level, right_ind]:
                # Left Explored First
                if self.heap_view[level, left_ind] > self.heap_view[level, i]:
                    temp_value = self.heap_view[level, i]
                    self.heap_view[level, i] = self.heap_view[level, left_ind]
                    self.heap_view[level, left_ind] = temp_value

                    temp_index = self.heap_inds_view[level, i]
                    self.heap_inds_view[level, i] = self.heap_inds_view[level, left_ind]
                    self.heap_inds_view[level, left_ind] = temp_index
                    i = left_ind
                else:
                    break

            else:
                # Right Explored First
                if self.heap_view[level, right_ind] > self.heap_view[level, i]:
                    temp_value = self.heap_view[level, i]
                    self.heap_view[level, i] = self.heap_view[level, right_ind]
                    self.heap_view[level, right_ind] = temp_value

                    temp_index = self.heap_inds_view[level, i]
                    self.heap_inds_view[level, i] = self.heap_inds_view[level, right_ind]
                    self.heap_inds_view[level, right_ind] = temp_index
                    i = right_ind
                else:
                    break

        return 0
