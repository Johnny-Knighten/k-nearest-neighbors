import numpy as np
import knn.distance_metrics as dm
import heapq
import math

from collections import Counter

#from knn.ball_tree_queries import query_ball_tree_cyth


# Wrapper Over heapq
class PriorityQueue:

    def __init__(self, is_min_heap=True):
        self.queue = []
        self.cnt = Counter()
        self.is_min_heap = is_min_heap

    def heappush(self, priority, value):
        self.cnt[priority] += 1
        q_priority = priority if self.is_min_heap else -1*priority
        heapq.heappush(self.queue, (q_priority, self.cnt[priority], value))

    def heappushpop(self, priority, value):
        self.cnt[priority] += 1
        q_priority = priority if self.is_min_heap else -1*priority
        heapq.heappushpop(self.queue, (q_priority, self.cnt[priority], value))

    def heappop(self):
        curr_top = heapq.heappop(self.queue)
        q_priority = curr_top[0] if self.is_min_heap else -1*curr_top[0]
        return q_priority, curr_top[2]

    def peektop(self):
        curr_top = self.queue[0]
        q_priority = curr_top[0] if self.is_min_heap else -1*curr_top[0]
        return q_priority, curr_top[2]


class KNNMixin:

    def _brute_force_knn(self, test_data, distances=False):
        distance_matrix = self.metric(self.train_data, test_data)
        k_smallest_ind = np.argpartition(distance_matrix, self.k-1)[:, :self.k]

        if distances:
            return k_smallest_ind, np.array([distance_matrix[i, x] for i, x in enumerate(k_smallest_ind)])

        return k_smallest_ind

    def _create_ball_tree(self, data, leaf_size, centroids, index):


        if data.shape[0] <= leaf_size:
            leaf_node = {}
            leaf_node["center"] = np.mean(data[:, 1:], axis=0)[np.newaxis, :]
            centroids[index, :] = leaf_node["center"] # New
            leaf_node["radius"] = np.max(self.metric(data[:, 1:], leaf_node["center"]))
            leaf_node["data"] = data
            return leaf_node

        # Random Point x0
        rand_index = np.random.choice(data.shape[0], 1, replace=False)
        rand_point = data[rand_index, 1:]

        # Find Maximal Point x1
        distances = self.metric(data[:, 1:], rand_point)
        ind_of_max_dist = np.argmax(distances)
        max_vector_1 = data[ind_of_max_dist, 1:]

        # Find Maximal Point x2
        distances = self.metric(data[:, 1:], max_vector_1[np.newaxis, :])
        ind_of_max_dist = np.argmax(distances)
        max_vector_2 = data[ind_of_max_dist, 1:]

        # Project Data
        proj_data = data[:, 1:].dot(max_vector_1-max_vector_2)

        # Find Median And Split Data
        median_ind = np.argpartition(proj_data, proj_data.size//2)
        lower_than_med_inds = median_ind[:proj_data.size//2]
        greater_than_med_inds = median_ind[proj_data.size//2:]

        # Create Circle
        center = np.mean(data[:, 1:], axis=0)
        radius = np.max(self.metric(data[:, 1:], center[np.newaxis, :]))

        left_index = 2 * index + 1
        right_index = left_index + 1

        internal_node = {}
        internal_node["center"] = center[np.newaxis, :]
        centroids[index, :] = internal_node["center"] # New
        internal_node["radius"] = radius
        internal_node["left_child"] = self._create_ball_tree(data[lower_than_med_inds], leaf_size, centroids, left_index)
        internal_node["right_child"] = self._create_ball_tree(data[greater_than_med_inds], leaf_size, centroids, right_index)

        return internal_node




class KNNClassification(KNNMixin):

    def __init__(self, k=1, metric="euclidean", tree=False, tree_leaf_size=1):
        self.k = k
        self.tree = tree
        self.tree_leaf_size = tree_leaf_size
        self.ball_tree = None

        if callable(metric):
            self.metric = metric
        else:
            metrics = {"euclidean": dm.euclidean,
                       "manhattan": dm.manhattan,
                       "hamming": dm.hamming,
                       "cosine": dm.cosine,
                       "pearson": dm.pearson,
                       "chisqr": dm.chisqr}
            # Default To Euclidean If Given Metric Does Not Exist
            self.metric = metrics.get(metric, dm.euclidean)

        self.labels = np.empty(0)
        self.train_data = np.empty(0)

    # Ensure Labels Are Type np.int
    def train(self, labels, train_data):
        self.labels = labels
        self.train_data = train_data

        # Build Tree - Not Implemented Yet
        if self.tree:
            numb_of_leafs = math.ceil(train_data.shape[0] / self.tree_leaf_size)
            tree_height = 1 + math.ceil(np.log2(numb_of_leafs))
            numb_of_nodes = int(2 ** tree_height) - 1
            self.centroids = np.zeros((numb_of_nodes, self.train_data.shape[1]))
            train_data_with_inds = np.hstack((np.arange(self.train_data.shape[0])[:, np.newaxis], self.train_data))
            self.ball_tree = self._create_ball_tree(train_data_with_inds, self.tree_leaf_size, self.centroids, 0)

    def predict(self, test_data):

        if self.tree:

            centroid_dists = self.metric(self.centroids, test_data)


            output_labels = []
            for i, test_vector in enumerate(test_data):
                queue = PriorityQueue(False)
                # Fill queue With High Distance Points
                list(map(lambda x: queue.heappush(9e10, np.array([9e10, 9e10])), range(self.k)))
                #query_ball_tree_cyth(test_vector[np.newaxis, :], self.k, queue, self.ball_tree, centroid_dists[i, :], 0, self.metric)
                nn_points = np.array([x[2][0] for x in queue.queue], dtype=np.int)

                predicted_labels = self.labels[nn_points]
                output_labels.append(np.bincount(predicted_labels).argmax())
            return np.array(output_labels)

        else:
            indices = self._brute_force_knn(test_data, False)
            labels = self.labels[indices]
            output_labels = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, labels)
            return output_labels


class KNNRegression(KNNMixin):

    def __init__(self, k=1, metric="euclidean", tree=False, tree_leaf_size=1):
        self.k = k
        self.tree = tree
        self.tree_leaf_size = tree_leaf_size
        self.ball_tree = None

        if callable(metric):
            self.metric = metric
        else:
            metrics = {"euclidean": dm.euclidean,
                       "manhattan": dm.manhattan,
                       "hamming": dm.hamming,
                       "cosine": dm.cosine,
                       "pearson": dm.pearson,
                       "chisqr": dm.chisqr}
            # Default To Euclidean If Given Metric Does Not Exist
            self.metric = metrics.get(metric, dm.euclidean)

        self.train_response = np.empty(0)
        self.train_data = np.empty(0)

    def train(self, train_response, train_data):
        self.train_response = train_response
        self.train_data = train_data

        # Build Tree - Not Implemented Yet
        if self.tree:
            self.ball_tree = self._create_ball_tree(train_data, self.tree_leaf_size)

    def predict(self, test_data):

        if self.tree:
            centroid_dists = self.metric(self.centroids, test_data)
            output_responses = []
            for i, test_vector in enumerate(test_data):
                queue = PriorityQueue(False)
                # Fill queue With High Distance Points
                list(map(lambda x: queue.heappush(9e10, np.array([9e10, 9e10])), range(self.k)))
                self._query_ball_tree(test_vector[np.newaxis, :], self.k, queue, self.ball_tree, centroid_dists[i, :], 0, self.metric)
                nn_points = np.array([x[2] for x in queue.queue])[:, np.newaxis]
                predicted_responses = self.train_response[np.where((self.train_data == nn_points).all(-1))[1]]
                output_responses.append(np.mean(predicted_responses))
            return np.array(output_responses)

        else:
            indices = self._brute_force_knn(test_data, False)
            responses = self.train_response[indices]
            output_responses = np.mean(responses, axis=1)
            return output_responses
