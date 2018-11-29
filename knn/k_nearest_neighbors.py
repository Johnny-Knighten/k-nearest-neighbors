import numpy as np
from knn.ball_tree import BallTree
import knn.distance_metrics as dm


class NNSearchMixin:

    def __init__(self):
        self.metric = None
        self.ball_tee = None

    # Brute Force Nearest Neighbor Search
    def _brute_force_nn_query(self, train_data, test_data, k=1, metric="euclidean", distances=False):

        # Get Desired Metric
        if callable(metric):
            self.metric = metric
        else:
            metrics = {"euclidean": dm.euclidean_pairwise,
                       "manhattan": dm.manhattan_pairwise,
                       "hamming": dm.hamming_pairwise,
                       "cosine": dm.cosine_pairwise,
                       "pearson": dm.pearson_pairwise,
                       "chisqr": dm.chisqr_pairwise}
            # Default To Euclidean If Given Metric Does Not Exist
            self.metric = metrics.get(metric, dm.euclidean_pairwise)

        distance_matrix = self.metric(train_data, test_data)
        k_smallest_ind = np.argpartition(distance_matrix, k-1)[:, :k]

        # Allows Distances To Be Returned
        if distances:
            return k_smallest_ind, np.array([distance_matrix[i, x] for i, x in enumerate(k_smallest_ind)])

        return k_smallest_ind

    # Build Ball Tree For Nearest Neighbor Search
    def _ball_tree_build(self, train_data, leaf_size=1, metric="euclidean"):
        self.ball_tree = BallTree(train_data, leaf_size, metric)
        self.ball_tree.build_tree()

    # Use Ball Tree To Perform Nearest Neighbor Search
    def _ball_tree_nn_query(self, query_vectors, k=1, distances=False):
        self.ball_tree.query(query_vectors, k)

        # Allows Distances To Be Returned
        if distances:
            return self.ball_tree.heap_inds, self.ball_tree.heap

        return self.ball_tree.heap_inds


class KNNClassification(NNSearchMixin):

    def __init__(self, metric="euclidean", use_tree=False, tree_leaf_size=1):
        self.use_tree = use_tree
        self.tree_leaf_size = tree_leaf_size
        self.metric = metric
        self.labels = np.empty(0)
        self.train_data = np.empty(0)

        super().__init__()

    # Ensure Labels Are Type np.int
    def train(self, labels, train_data):
        self.labels = labels
        self.train_data = train_data

        # Build Tree - Not Implemented Yet
        if self.use_tree:
            self._ball_tree_build(train_data, self.tree_leaf_size, self.metric)

    def predict(self, test_data, k):

        if self.use_tree:
            indices = self._ball_tree_nn_query(test_data, k)
        else:
            indices = self._brute_force_nn_query(self.train_data, test_data, k, self.metric, False)

        labels = self.labels[indices]
        # Find The Most Frequently Assigned Label For Each Test Vector(Row)
        output_labels = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, labels)
        return output_labels


class KNNRegression(NNSearchMixin):

    def __init__(self, metric="euclidean", use_tree=False, tree_leaf_size=1):
        self.use_tree = use_tree
        self.tree_leaf_size = tree_leaf_size
        self.metric = metric
        self.train_response = np.empty(0)
        self.train_data = np.empty(0)

        super().__init__()

    def train(self, train_response, train_data):
        self.train_response = train_response
        self.train_data = train_data

        if self.use_tree:
            self._ball_tree_build(train_data, self.tree_leaf_size, self.metric)

    def predict(self, test_data, k):

        if self.use_tree:
            indices = self._ball_tree_nn_query(test_data, k)
        else:
            indices = self._brute_force_nn_query(self.train_data, test_data, k, self.metric, False)

        responses = self.train_response[indices]
        output_responses = np.mean(responses, axis=1)
        return output_responses
