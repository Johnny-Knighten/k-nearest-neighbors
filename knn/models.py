import numpy as np

from knn.mixins.nnsearch import NNSearchMixin


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
            indices, _ = self._ball_tree_nn_query(test_data, k)
        else:
            indices, _ = self._brute_force_nn_query(self.train_data, test_data, k, self.metric)

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
            indices, _ = self._ball_tree_nn_query(test_data, k)
        else:
            indices, _ = self._brute_force_nn_query(self.train_data, test_data, k, self.metric)

        responses = self.train_response[indices]
        output_responses = np.mean(responses, axis=1)
        return output_responses
