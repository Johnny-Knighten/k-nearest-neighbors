import cython
import numpy as np
cimport numpy as np
import math
from knn.distance_metrics import euclidean

cdef class BallTree:

    cdef readonly double[:, ::1] data_view
    cdef readonly long[::1] data_inds_view
    cdef np.ndarray data
    cdef np.ndarray data_inds

    cdef public np.ndarray node_data_inds
    cdef public np.ndarray node_radius
    cdef public np.ndarray node_is_leaf
    cdef public np.ndarray node_center
    cdef public long[:,::1] node_data_inds_view
    cdef public double[::1] node_radius_view
    cdef public double[::1] node_is_leaf_view
    cdef public double[:,::1] node_center_view

    cdef int leaf_size
    cdef int node_count
    cdef int tree_height

    def __init__(self, data, leaf_size):

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


    def build_tree(self):
        self._build(0, 0, self.data.shape[0]-1)


    def _build(self, long node_index, long node_data_start, long node_data_end):

        ##########################
        # Current Node Is A Leaf #
        ##########################
        if (node_data_end-node_data_start+1) <= self.leaf_size:

            self.node_center[node_index] = np.mean(self.data[self.data_inds[node_data_start:node_data_end+1]], axis=0)

            self.node_radius[node_index] = np.max(euclidean(self.data[self.data_inds[node_data_start:node_data_end+1]],
                                                            self.node_center[node_index, :][np.newaxis, :]))

            self.node_data_inds[node_index, 0] = node_data_start
            self.node_data_inds[node_index, 1] = node_data_end

            self.node_is_leaf[node_index] = True

            return None

        #################################
        # Current Node Is Internal Node #
        #################################

        # Select Random Point
        rand_index = np.random.choice(node_data_end-node_data_start+1, 1, replace=False)
        rand_point = self.data[self.data_inds[rand_index], :]

        # Find Point Max Distance From Random Point
        distances = euclidean(self.data[self.data_inds[node_data_start:node_data_end+1]], rand_point)
        ind_of_max_dist = np.argmax(distances)
        max_vector_1 = self.data[ind_of_max_dist]

        # Find Point Max Distance From Previous Point
        distances = euclidean(self.data[self.data_inds[node_data_start:node_data_end+1]], max_vector_1[np.newaxis, :])
        ind_of_max_dist = np.argmax(distances)
        max_vector_2 = self.data[ind_of_max_dist]

        # Project Data On Vector Between Previous Two Points
        proj_data = np.dot(self.data[self.data_inds[node_data_start:node_data_end+1]], max_vector_1-max_vector_2)


        # Find Median
        median = np.partition(proj_data, proj_data.size//2)[proj_data.size//2]

        # Partition Data Around Median - Currently Using Hoare Partitioning
        low = node_data_start
        high = node_data_end
        pivot = median
        self._hoare_partition(pivot, low, high, proj_data)

        # Set Info About Current Ball
        center = np.mean(self.data[self.data_inds[node_data_start:node_data_end+1]], axis=0)
        radius = np.max(euclidean(self.data[self.data_inds[node_data_start:node_data_end+1]], center[np.newaxis, :]))
        self.node_radius[node_index] = radius
        self.node_center[node_index] = center
        self.node_data_inds[node_index, 0] = node_data_start
        self.node_data_inds[node_index, 1] = node_data_end
        self.node_is_leaf[node_index] = False

        # Keep Generating Tree
        left_index = 2 * node_index + 1
        right_index = left_index + 1
        self._build(left_index, node_data_start,  node_data_start+ (proj_data.size//2)-1 )
        self._build(right_index, node_data_start+(proj_data.size//2),   node_data_end)


    def _hoare_partition(self, pivot, low, high, projected_data):

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
            if i >= j:
                return None

            # Swap Values
            projected_data[i2], projected_data[j2] = projected_data[j2], projected_data[i2]
            self.data_inds[i], self.data_inds[j] = self.data_inds[j], self.data_inds[i]
