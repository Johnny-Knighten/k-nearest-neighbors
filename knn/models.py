import numpy as np
from knn.ball_tree import BallTree
import knn.distance_metrics as dm

class NNSearchMixin:
    """ A Mixin that provides methods for brute force NN queries, ball tree construction, and ball tree queries.

    This mixin is used to provide foundational methods that are required by all NN models. All distance metrics used
    are made using Cython. Brute Force NN offers more distance metric options than the Ball Tree methods and allows
    the user to pass their own function as the metric.

    Could be used as a standalone object, but it is intended to be used as a mixin for other classes.

    Attributes:
        metric (string or callable): A string  or callable representing the metric to be used by the NN methods.
            Defaults to euclidean distance.
        ball_tree (:obj:'BallTree'): The Ball Tree constructed and used by the Ball Tree methods

    """

    def __init__(self):
        """ Creates a NNSearchMixin object.

        """
        self.metric = None
        self.ball_tee = None

    def _brute_force_nn_query(self, train_data, test_data, k=1, metric="euclidean", distances=False):
        """ Finds the NNs of a set of query vectors using brute force.

        Args:
            train_data(ndarray): A 2D array of vectors being searched through.
            test_data(ndarray): A 2D array of query vectors.
            k(int): The number of NNs to be found.
            metric(string or callable): The metric used in the search.
            distances(bool): A flag to determine if distances will be returned.

        Returns:
            (tuple): tuple containing:
                k_smallest_ind (ndarray): A 2D array of of the indices of the NNs for each query vector.
                distances (ndarray): A 2D array of distances of the NNs for each query vector.

        """

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

    def _ball_tree_build(self, train_data, leaf_size=1, metric="euclidean"):
        """ Creates A Ball Tree using the supplied vectors.

        Args:
            train_data(ndarray): A 2D array of vectors being searched through.
            leaf_size(int): The number of vectors contained in the leaves of the Ball Tree.
            metric(string): The metric used in the search.

        """
        self.ball_tree = BallTree(train_data, leaf_size, metric)
        self.ball_tree.build_tree()

    def _ball_tree_nn_query(self, query_vectors, k=1, distances=False):
        """ Peforms a NN search using A Ball Tree.

        Must execute _ball_tree_build() before using this method.

        Args:
            query_vectors(ndarray): A 2D array of query vectors.
            k(int): The number of NNs to be found.
            distances(bool): A flag to determine if distances will be returned.

        Returns:
            (tuple): tuple containing:
                k_smallest_ind (ndarray): A 2D array of of the indices of the NNs for each query vector.
                distances (ndarray): A 2D array of distances of the NNs for each query vector.


        """
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
