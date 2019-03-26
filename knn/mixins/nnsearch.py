import numpy as np
import knn.distance_metrics as dm

from knn.ball_tree import BallTree


class NNSearchMixin:
    """ A Mixin that provides methods for brute force NN queries, ball tree construction, and ball tree queries.

    This mixin is used to provide foundational methods that are required by all NN models. All distance metrics used
    are made using Cython. Brute Force NN offers more distance metric options than the Ball Tree methods and allows
    the user to pass their own function as the metric.

    Could be used as a standalone object, but it is intended to be used as a mixin for other classes.

    Attributes:
        ball_tree (:obj:'BallTree'): The Ball Tree constructed and used by the Ball Tree methods
        numb_of_train_vectors(int): The number of training vectors

    """

    def __init__(self):
        """ Creates a NNSearchMixin object.

        """
        self.ball_tee = None
        self.number_of_train_vectors = 0

    def _brute_force_nn_query(self, train_data, test_data, k=1, metric="euclidean"):
        """ Finds the NNs of a set of query vectors using brute force.

        NN are not sorted by smallest distance.

        Args:
            train_data(ndarray): A 2D array of vectors being searched through.
            test_data(ndarray): A 2D array of query vectors.
            k(int): The number of NNs to be found.
            metric(string or callable): The metric used in the search.

        Returns:
            (tuple): tuple containing:
                k_smallest_indices (ndarray): A 2D array of the indices of the k NNs for each query vector.
                k_smallest_distances (ndarray): A 2D array of distances of the NNs for each query vector.

        """
        if k > train_data.shape[0]:
            raise ValueError("k Must Be Smaller Than Or Equal To The Number Of Training Vectors")

        if k < 0:
            raise ValueError("k Must Be Greater Than 0")

        if not isinstance(k, int):
            raise ValueError("k Must Be An Integer")


        # The User Is Allowed To Pass In Their Own Metric
        if not callable(metric):
            metrics = {"euclidean": dm.euclidean_pairwise,
                       "manhattan": dm.manhattan_pairwise,
                       "hamming": dm.hamming_pairwise,
                       "cosine": dm.cosine_pairwise,
                       "pearson": dm.pearson_pairwise,
                       "chisqr": dm.chisqr_pairwise}

            metric = metrics.get(metric, dm.euclidean_pairwise)

        distance_matrix = metric(train_data, test_data)
        k_smallest_indices = np.argpartition(distance_matrix, k-1)[:, :k]
        k_smallest_distances = np.array([distance_matrix[i, x] for i, x in enumerate(k_smallest_indices)])

        return k_smallest_indices, k_smallest_distances

    def _ball_tree_build(self, train_data, leaf_size=1, metric="euclidean"):
        """ Creates A Ball Tree using the supplied vectors.

        Args:
            train_data(ndarray): A 2D array of vectors being searched through.
            leaf_size(int): The number of vectors contained in the leaves of the Ball Tree.
            metric(string): The metric used in the search.

        """
        self.number_of_train_vectors = train_data.shape[0]
        self.ball_tree = BallTree(train_data, leaf_size, metric)
        self.ball_tree.build_tree()

    def _ball_tree_nn_query(self, query_vectors, k=1):
        """ Performs a NN search using A Ball Tree.

        Must execute _ball_tree_build() before using this method. NN are not sorted by smallest distance.

        Args:
            query_vectors(ndarray): A 2D array of query vectors.
            k(int): The number of NNs to be found.

        Returns:
            (tuple): tuple containing:
                k_smallest_ind (ndarray): A 2D array of of the indices of the NNs for each query vector.
                distances (ndarray): A 2D array of distances of the NNs for each query vector.

        """
        if k > self.number_of_train_vectors:
            raise ValueError("k Must Be Smaller Than Or Equal To The Number Of Training Vectors")

        if k < 0:
            raise ValueError("k Must Be Greater Than 0")
        
        if not isinstance(k, int):
            raise ValueError("k Must Be An Integer")

        self.ball_tree.query(query_vectors, k)

        return self.ball_tree.heap_inds, self.ball_tree.heap
