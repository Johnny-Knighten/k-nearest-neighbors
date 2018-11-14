import numpy as np
import knn.distance_metrics as dm
from knn.ball_tree import BallTree
import math
import knn.distance_metrics_cython as dmc


class KNNMixin:

    def _brute_force_knn(self, test_data, metric="euclidean", distances=False, ):

        # Get Desired Metric
        if callable(metric):
            self.metric = metric
        else:
            metrics = {"euclidean": dmc.euclidean_pairwise,
                       "manhattan": dmc.manhattan_pairwise,
                       "hamming": dmc.hamming_pairwise,
                       "cosine": dm.cosine,
                       "pearson": dm.pearson,
                       "chisqr": dm.chisqr}
            # Default To Euclidean If Given Metric Does Not Exist
            self.metric = metrics.get(metric, dm.euclidean)

        distance_matrix = self.metric(self.train_data, test_data)
        k_smallest_ind = np.argpartition(distance_matrix, self.k-1)[:, :self.k]

        if distances:
            return k_smallest_ind, np.array([distance_matrix[i, x] for i, x in enumerate(k_smallest_ind)])

        return k_smallest_ind

    def _train_tree(self, train_data, leaf_size, metric="euclidean"):
        ball_tree = BallTree(train_data, leaf_size, metric)
        ball_tree.build_tree()
        return ball_tree

    def _query_tree(self, ball_tree, query_vectors):
        ball_tree.query(query_vectors, self.k)
        return ball_tree.heap_inds





class KNNClassification(KNNMixin):

    def __init__(self, k=1, metric="euclidean", use_tree=False, tree_leaf_size=1):
        self.k = k
        self.use_tree = use_tree
        self.tree_leaf_size = tree_leaf_size
        self.ball_tree = None
        self.metric = metric
        self.labels = np.empty(0)
        self.train_data = np.empty(0)

    # Ensure Labels Are Type np.int
    def train(self, labels, train_data):
        self.labels = labels
        self.train_data = train_data

        # Build Tree - Not Implemented Yet
        if self.use_tree:
            self.ball_tree = self._train_tree(train_data, self.tree_leaf_size, self.metric)

    def predict(self, test_data):

        if self.use_tree:
            indices = self._query_tree(self.ball_tree, test_data)
            labels = self.labels[indices]
            output_labels = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, labels)
            return output_labels

        else:
            indices = self._brute_force_knn(test_data, self.metric, False)
            labels = self.labels[indices]
            output_labels = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, labels)
            return output_labels


class KNNRegression(KNNMixin):

    def __init__(self, k=1, metric="euclidean", use_tree=False, tree_leaf_size=1):
        self.k = k
        self.use_tree = use_tree
        self.tree_leaf_size = tree_leaf_size
        self.ball_tree = None
        self.metric = metric
        self.train_response = np.empty(0)
        self.train_data = np.empty(0)

    def train(self, train_response, train_data):
        self.train_response = train_response
        self.train_data = train_data

        # Build Tree - Not Implemented Yet
        if self.use_tree:
            self.ball_tree = self._train_tree(train_data, self.tree_leaf_size, self.metric)

    def predict(self, test_data):

        if self.use_tree:
            indices = self._query_tree(self.ball_tree, test_data)
            responses = self.train_response[indices]
            output_responses = np.mean(responses, axis=1)
            return output_responses

        else:
            indices = self._brute_force_knn(test_data, self.metric, False)
            responses = self.train_response[indices]
            output_responses = np.mean(responses, axis=1)
            return output_responses
