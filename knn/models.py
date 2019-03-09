import numpy as np

from knn.mixins.nnsearch import NNSearchMixin


class KNNClassification(NNSearchMixin):
    """ KNN model that performs classification.

    Attributes:
        use_tree(bool): Flag used to pick if a Ball Tree will be used. Default(False) is to use the Brute Force
            algorithm.
        tree_leaf_size(int): The number of vectors contained in the leaves of the Ball Tree.
        metric(string or callable): The distance metric used in the search.
        train_labels(ndarray): A 1D array of numeric labels of the training data. The Y data.
        train_data(ndarray): A 2D array of training data. The X data.

    """

    def __init__(self, metric="euclidean", use_tree=False, tree_leaf_size=1):
        """ Creates a KNN model for classification.

        If you are using the Brute Force algorithm you can supply your own distance metric. This function needs
        to return a distance matrix whose rows represent the testing data and columns represent the testing data. If
        you are not using your own distance metric supply the metric param with one of the following strings:
        euclidean, manhattan, hamming, cosine, pearson, and chisqr.

        The Brute Force algorithm has six distance metrics available: euclidean, manhattan, hamming, cosine, pearson,
        and chi-squared.

        When using a Ball Tree only three distance metrics are available: euclidean, manhattan, and hamming.

        Args:
            metric(string or callable): The distance metric used in the search.
            use_tree(bool): Flag used to pick if a Ball Tree will be used. Default(False) is to use the Brute Force
            algorithm.
            tree_leaf_size:(int): The number of vectors contained in the leaves of the Ball Tree.

        """
        self.use_tree = use_tree
        self.tree_leaf_size = tree_leaf_size
        self.metric = metric
        self.train_labels = np.empty(0)
        self.train_data = np.empty(0)

        super().__init__()

    # Ensure Labels Are Type np.int
    def train(self, train_labels, train_data):
        """ Trains the model.

        If Brute Force KNN is being used then the training data and responses are simply stored. If a Ball Tree is being
        used then the Ball Tree is constructed.

        Args:
            train_labels(ndarray): A 1D array of numeric labels of the training data. The Y data.
            train_data(ndarray): A 2D array of training data. The X data.

        """

        self.train_labels = train_labels
        self.train_data = train_data

        if self.use_tree:
            self._ball_tree_build(train_data, self.tree_leaf_size, self.metric)

    def predict(self, test_data, k):
        """Performs classification on the supplied test data.

        Args:
            test_data(ndarray):  A 2D array of vectors whose labels are to be predicted.
            k(int): The number of NNs used to make a prediction.

        Returns:
            (ndarray): A 1D array of predicted classifications. Predicted Y values.

        """

        if self.use_tree:
            indices, _ = self._ball_tree_nn_query(test_data, k)
        else:
            indices, _ = self._brute_force_nn_query(self.train_data, test_data, k, self.metric)

        labels = self.train_labels[indices]
        # Find The Most Frequently Assigned Label For Each Test Vector(Row)
        output_labels = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, labels)
        return output_labels


class KNNRegression(NNSearchMixin):
    """ KNN that performs regression.

    Attributes:
        use_tree(bool): Flag used to pick if a Ball Tree will be used. Default(False) is to use the Brute Force
            algorithm.
        tree_leaf_size(int): The number of vectors contained in the leaves of the Ball Tree. Defaults to one vector.
        metric(string or callable): The distance metric used in the search.
        train_response(ndarray): A 1D array of the training data's responses. The Y values.
        train_data(ndarray): A 2D array of training data. The X values.

    """

    def __init__(self, metric="euclidean", use_tree=False, tree_leaf_size=1):
        """ Creates a KNN model for regression.

        If you are using the Brute Force algorithm you can supply your own distance metric. This function needs
        to return a distance matrix whose rows represent the testing data and columns represent the testing data. If
        you are not using your own distance metric supply the metric param with one of the following strings:
        euclidean, manhattan, hamming, cosine, pearson, and chisqr.

        The Brute Force algorithm has six distance metrics available: euclidean, manhattan, hamming, cosine, pearson,
        and chi-squared.

        When using a Ball Tree only three distance metrics are available: euclidean, manhattan, and hamming.

        Args:
            metric(string or callable): The distance metric used in the search.
            use_tree(bool): Flag used to pick if a Ball Tree will be used. Default(False) is to use the Brute Force
            algorithm.
            tree_leaf_size:(int): The number of vectors contained in the leaves of the Ball Tree.

        """
        self.use_tree = use_tree
        self.tree_leaf_size = tree_leaf_size
        self.metric = metric
        self.train_response = np.empty(0)
        self.train_data = np.empty(0)

        super().__init__()

    def train(self, train_response, train_data):
        """ Trains the model.

        If Brute Force KNN is being used than the training data and responses are simply stored. If a Ball Tree is being
        used then the Ball Tree is constructed.

        Args:
            train_response(ndarray): A 1D array of the training data's responses.
            train_data(ndarray): A 2D array of training data.

        """
        self.train_response = train_response
        self.train_data = train_data

        if self.use_tree:
            self._ball_tree_build(train_data, self.tree_leaf_size, self.metric)

    def predict(self, test_data, k):
        """ Performs regressions on the supplied test data.

        Args:
            test_data(ndarray):  A 2D array of vectors whose responses are to be predicted.
            k(int): The number of NNs used to make a prediction.

        Returns:
            (ndarray): A 1D array of predicted responses.

        """
        if self.use_tree:
            indices, _ = self._ball_tree_nn_query(test_data, k)
        else:
            indices, _ = self._brute_force_nn_query(self.train_data, test_data, k, self.metric)

        responses = self.train_response[indices]
        output_responses = np.mean(responses, axis=1)
        return output_responses
